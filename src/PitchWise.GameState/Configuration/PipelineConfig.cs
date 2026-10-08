using Microsoft.Extensions.Logging;
using PitchWise.GameState.Calibration;
using PitchWise.GameState.Detection;
using PitchWise.GameState.Jersey;
using PitchWise.GameState.Reid;
using PitchWise.GameState.Team;
using PitchWise.GameState.Track;
using StrongSort;
using PitchWise.Tracking.Pipeline;
using PitchWise.Tracking.Wrappers;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace PitchWise.GameState.Configuration;

/// <summary>
/// .NET counterpart of the Hydra soccernet.yaml: declarative module selection and
/// hyperparameters. Modules with enabled=auto activate when their ONNX models exist.
/// </summary>
/// <remarks>
/// Lives beside the modules it assembles rather than in any one executable: the module order
/// is the product, and every consumer (CLI, desktop app, a future streaming backend) needs the
/// same one. A copy per host would drift.
/// </remarks>
public sealed class PipelineConfig
{
    public string ModelsDir { get; set; } = "pretrained_models";
    public OnnxConfig Onnx { get; set; } = new();
    public DetectorConfig BboxDetector { get; set; } = new();
    public ReidConfig Reid { get; set; } = new();
    public TrackConfig Track { get; set; } = new();
    public PitchConfig Pitch { get; set; } = new();
    public CalibrationConfig Calibration { get; set; } = new();
    public JerseyConfig Jersey { get; set; } = new();

    public sealed class OnnxConfig
    {
        public string Provider { get; set; } = "cpu";
        public int DeviceId { get; set; }
    }

    public sealed class DetectorConfig
    {
        public string Model { get; set; } = "yolo11m.onnx";
        public float MinConfidence { get; set; } = 0.4f;
        public int InputSize { get; set; } = 640;
    }

    public sealed class ReidConfig
    {
        public string Enabled { get; set; } = "auto";
        public string Model { get; set; } = "prtreid-soccernet-baseline.onnx";
        public int BatchSize { get; set; } = 8;
    }

    public sealed class TrackConfig
    {
        public float EmaAlpha { get; set; } = 0.9f;
        public float McLambda { get; set; } = 0.995f;
        public float MaxAppearanceDistance { get; set; } = 0.5f;
        public float MaxIouDistance { get; set; } = 0.8f;
        public int MaxAge { get; set; } = 300;
        public int NInit { get; set; } = 0;
        public float MinConfidence { get; set; } = 0f;
    }

    public sealed class PitchConfig
    {
        public string Enabled { get; set; } = "auto";
        public string KpModel { get; set; } = "nbjw_kp.onnx";
        public string LinesModel { get; set; } = "nbjw_lines.onnx";
    }

    public sealed class CalibrationConfig
    {
        public string Enabled { get; set; } = "auto";
        public bool UsePrevHomography { get; set; } = true;
    }

    public sealed class JerseyConfig
    {
        public string Enabled { get; set; } = "auto";
        public string DetModel { get; set; } = "mmocr_det.onnx";
        public string RecModel { get; set; } = "mmocr_rec.onnx";
        public string Charset { get; set; } = "mmocr_charset.txt";
    }

    public static PipelineConfig Load(string? path)
    {
        if (path is null || !File.Exists(path))
            return new PipelineConfig();
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();
        return deserializer.Deserialize<PipelineConfig>(File.ReadAllText(path));
    }

    /// <summary>Builds the tracking pipeline for one video resolution.</summary>
    /// <remarks>
    /// Sets the static OnnxSessions.Provider/DeviceId as a side effect, so two pipelines with
    /// different providers must not be built concurrently.
    /// </remarks>
    public Pipeline BuildPipeline(int imWidth, int imHeight, ILogger? log = null)
    {
        string Model(string name) => Path.Combine(ModelsDir, name);
        bool Exists(params string[] names) => names.All(n => File.Exists(Model(n)));
        bool On(string enabled, params string[] models)
            => enabled.ToLowerInvariant() switch
            {
                "true" or "yes" or "on" => true,
                "false" or "no" or "off" => false,
                _ => Exists(models),
            };

        OnnxSessions.Provider = Onnx.Provider.ToLowerInvariant() switch
        {
            "dml" or "directml" => OnnxProvider.DirectML,
            "cuda" => OnnxProvider.Cuda,
            _ => OnnxProvider.Cpu,
        };
        OnnxSessions.DeviceId = Onnx.DeviceId;
        OnnxSessions.Log = log;
        log?.LogInformation("ONNX provider: {Provider}", OnnxSessions.Provider);

        if (!Exists(BboxDetector.Model))
            // the full path, not the configured one: models_dir is usually relative, and which
            // directory it resolved against is exactly what is in doubt when this fires
            throw new FileNotFoundException(
                $"missing {Path.GetFullPath(Model(BboxDetector.Model))} — export it with " +
                "tools/export/export_yolo.py, or point models_dir at an existing export");

        var hasReid = On(Reid.Enabled, Reid.Model);
        var hasCalib = On(Pitch.Enabled, Pitch.KpModel, Pitch.LinesModel)
                       && On(Calibration.Enabled, Pitch.KpModel, Pitch.LinesModel);
        var hasOcr = On(Jersey.Enabled, Jersey.DetModel, Jersey.RecModel, Jersey.Charset);

        var modules = new List<Module>
        {
            new YoloOnnxDetector(Model(BboxDetector.Model), BboxDetector.MinConfidence,
                inputSize: BboxDetector.InputSize),
        };
        if (hasReid)
            modules.Add(new PrtReidOnnx(Model(Reid.Model), Reid.BatchSize));
        modules.Add(new StrongSortModule(new StrongSortOptions
        {
            EmaAlpha = Track.EmaAlpha,
            McLambda = Track.McLambda,
            MaxAppearanceDistance = Track.MaxAppearanceDistance,
            MaxIouDistance = Track.MaxIouDistance,
            MaxAge = Track.MaxAge,
            NInit = Track.NInit,
            MinConfidence = Track.MinConfidence,
        }));
        if (hasCalib)
        {
            modules.Add(new NbjwCalibKeypoints(Model(Pitch.KpModel), Model(Pitch.LinesModel)));
            modules.Add(new NbjwCalib(imWidth, imHeight, Calibration.UsePrevHomography));
        }
        if (hasOcr)
            modules.Add(new MmocrOnnx(Model(Jersey.DetModel), Model(Jersey.RecModel), Model(Jersey.Charset)));

        var voteAttributes = new List<string>();
        if (hasOcr) voteAttributes.Add("jersey_number");
        if (hasReid) voteAttributes.Add("role");
        if (voteAttributes.Count > 0)
            modules.Add(new MajorityVoteTracklet(voteAttributes.ToArray()));
        if (hasReid)
        {
            modules.Add(new TrackletTeamClustering());
            if (hasCalib)
                modules.Add(new TrackletTeamSideLabeling());
        }

        log?.LogInformation("reid={Reid} calibration={Calib} ocr={Ocr}", hasReid, hasCalib, hasOcr);
        return new Pipeline(modules, log);
    }
}
