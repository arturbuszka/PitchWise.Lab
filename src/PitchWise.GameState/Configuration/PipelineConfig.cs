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
    public TrackletMergingConfig TrackletMerging { get; set; } = new();
    public PitchConfig Pitch { get; set; } = new();
    public CalibrationConfig Calibration { get; set; } = new();
    public JerseyConfig Jersey { get; set; } = new();
    public RoleCorrectionConfig RoleCorrection { get; set; } = new();
    public TeamConfig Team { get; set; } = new();

    public sealed class OnnxConfig
    {
        public string Provider { get; set; } = "cpu";
        public int DeviceId { get; set; }
        /// <summary>CUDA on small GPUs (≤ 6 GB): release arena memory between runs, see OnnxSessions.LowVram.</summary>
        public bool LowVram { get; set; }
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
        public int MaxKalmanPredictionsWithoutUpdate { get; set; } = 7;
        /// <summary>bpbreid (tracklab, aspect-ratio noise scales with box height) | strongsort (original paper constants).</summary>
        public string KalmanNoise { get; set; } = "bpbreid";
        public bool GatingUsesConfidence { get; set; } = false;
        /// <summary>Camera motion compensation: none (sn-gamestate) | ecc (tracklab ecc.py) | flow (sparse optical flow).</summary>
        public string CameraMotion { get; set; } = "none";

        public CameraMotionMode ParsedCameraMotion => CameraMotion.ToLowerInvariant() switch
        {
            "none" => CameraMotionMode.None,
            "ecc" => CameraMotionMode.Ecc,
            "flow" => CameraMotionMode.Flow,
            _ => throw new ArgumentException($"track.camera_motion: unknown value '{CameraMotion}' (expected none, ecc or flow)"),
        };

        public StrongSort.KalmanNoise ParsedKalmanNoise => KalmanNoise.ToLowerInvariant() switch
        {
            "bpbreid" => StrongSort.KalmanNoise.BpbReid,
            "strongsort" => StrongSort.KalmanNoise.StrongSort,
            _ => throw new ArgumentException($"track.kalman_noise: unknown value '{KalmanNoise}' (expected bpbreid or strongsort)"),
        };
    }

    /// <summary>Offline tracklet merging (<see cref="GameState.Track.TrackletMerging"/>); off unless enabled.</summary>
    public sealed class TrackletMergingConfig
    {
        public bool Enabled { get; set; }
        public double MaxGapSeconds { get; set; } = 10;
        public double MaxDistanceMeters { get; set; } = 3;
        public double MaxSpeed { get; set; } = 8;
        public double MaxAppearanceDistance { get; set; } = 0.3;
        public double DistanceWeight { get; set; } = 0.3;
        public double GapWeight { get; set; } = 0.1;
        public double NoLinkCost { get; set; } = 0.1;
        public int MinTeamDetections { get; set; } = 10;

        public TrackletMergingOptions ToOptions() => new()
        {
            MaxGapSeconds = MaxGapSeconds,
            MaxDistanceMeters = MaxDistanceMeters,
            MaxSpeed = MaxSpeed,
            MaxAppearanceDistance = MaxAppearanceDistance,
            DistanceWeight = DistanceWeight,
            GapWeight = GapWeight,
            NoLinkCost = NoLinkCost,
            MinTeamDetections = MinTeamDetections,
        };
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

    /// <summary>Role correction by jersey colour (<see cref="TrackletRoleCorrection"/>); off unless enabled.</summary>
    public sealed class RoleCorrectionConfig
    {
        public bool Enabled { get; set; }
        public double MaxRefereesPerFrame { get; set; } = 3;
        public double MaxGoalkeepersPerFrame { get; set; } = 2;
        public double LightnessWeight { get; set; } = 0.25;
        public double TeamColorRadius { get; set; } = 20;
        public double GoalkeeperColorDistance { get; set; } = 15;
        public double GoalkeeperMinAbsX { get; set; } = 38;
        public int MinColorDetections { get; set; } = 10;

        public TrackletRoleCorrectionOptions ToOptions() => new()
        {
            MaxRefereesPerFrame = MaxRefereesPerFrame,
            MaxGoalkeepersPerFrame = MaxGoalkeepersPerFrame,
            LightnessWeight = LightnessWeight,
            TeamColorRadius = TeamColorRadius,
            GoalkeeperColorDistance = GoalkeeperColorDistance,
            GoalkeeperMinAbsX = GoalkeeperMinAbsX,
            MinColorDetections = MinColorDetections,
        };
    }

    public sealed class TeamConfig
    {
        /// <summary>What tracklets are split into teams by: "color" (jersey colour, TorsoColor) or
        /// "embedding" (mean ReID embedding, as the Python sn-gamestate).</summary>
        public string Feature { get; set; } = "color";

        public TeamFeature ParsedFeature => Feature.ToLowerInvariant() switch
        {
            "color" or "colour" => TeamFeature.Color,
            "embedding" or "embeddings" or "reid" => TeamFeature.Embedding,
            _ => throw new ArgumentException($"team.feature: unknown value '{Feature}' (expected color or embedding)"),
        };
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

    /// <summary>
    /// What each configurable module's output depends on (its config section, the ONNX provider
    /// and the size of its model files), keyed by module name. A state cache records it so a
    /// later run refuses to reuse outputs of a module whose fingerprint changed.
    /// </summary>
    public IReadOnlyDictionary<string, string> ModuleFingerprints()
    {
        string Fingerprint(object section, params string[] models) => System.Text.Json.JsonSerializer.Serialize(new
        {
            Section = section,
            Provider = models.Length > 0 ? Onnx.Provider : null,
            Models = models.Select(m => Path.Combine(ModelsDir, m))
                .Select(m => $"{Path.GetFileName(m)}:{(File.Exists(m) ? new FileInfo(m).Length : -1)}"),
        });
        return new Dictionary<string, string>
        {
            [nameof(YoloOnnxDetector)] = Fingerprint(BboxDetector, BboxDetector.Model),
            [nameof(PrtReidOnnx)] = Fingerprint(Reid, Reid.Model),
            [nameof(StrongSortModule)] = Fingerprint(Track),
            [nameof(GameState.Track.TrackletMerging)] = Fingerprint(TrackletMerging),
            [nameof(NbjwCalibKeypoints)] = Fingerprint(Pitch, Pitch.KpModel, Pitch.LinesModel),
            [nameof(NbjwCalib)] = Fingerprint(Calibration),
            [nameof(MmocrOnnx)] = Fingerprint(Jersey, Jersey.DetModel, Jersey.RecModel, Jersey.Charset),
            // no settings: the entry only makes a cache saved before the module existed count as stale
            [nameof(TorsoColor)] = Fingerprint(new { }),
            [nameof(TrackletRoleCorrection)] = Fingerprint(RoleCorrection),
            [nameof(TrackletTeamClustering)] = Fingerprint(Team),
        };
    }

    /// <summary>Builds the tracking pipeline for one video resolution and frame rate.</summary>
    /// <remarks>
    /// Sets the static OnnxSessions.Provider/DeviceId as a side effect, so two pipelines with
    /// different providers must not be built concurrently.
    /// </remarks>
    public Pipeline BuildPipeline(int imWidth, int imHeight, ILogger? log = null, double fps = 25)
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
        OnnxSessions.LowVram = Onnx.LowVram;
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
        var teamFeature = Team.ParsedFeature;
        if (hasReid)
            modules.Add(new PrtReidOnnx(Model(Reid.Model), Reid.BatchSize));
        if (hasReid && (teamFeature == TeamFeature.Color || TrackletMerging.Enabled || RoleCorrection.Enabled))
            modules.Add(new TorsoColor());
        modules.Add(new StrongSortModule(new StrongSortOptions
        {
            EmaAlpha = Track.EmaAlpha,
            McLambda = Track.McLambda,
            MaxAppearanceDistance = Track.MaxAppearanceDistance,
            MaxIouDistance = Track.MaxIouDistance,
            MaxAge = Track.MaxAge,
            NInit = Track.NInit,
            MinConfidence = Track.MinConfidence,
            MaxKalmanPredictionsWithoutUpdate = Track.MaxKalmanPredictionsWithoutUpdate,
            KalmanNoise = Track.ParsedKalmanNoise,
            GatingUsesConfidence = Track.GatingUsesConfidence,
        }, Track.ParsedCameraMotion));
        if (hasCalib)
        {
            modules.Add(new NbjwCalibKeypoints(Model(Pitch.KpModel), Model(Pitch.LinesModel)));
            modules.Add(new NbjwCalib(imWidth, imHeight, Calibration.UsePrevHomography));
        }
        if (TrackletMerging.Enabled)
        {
            if (!hasReid || !hasCalib)
                throw new InvalidOperationException(
                    "tracklet_merging needs reid (embeddings, role_detection) and calibration (bbox_pitch)");
            modules.Add(new TrackletMerging(TrackletMerging.ToOptions(), fps, log));
        }
        if (hasOcr)
            modules.Add(new MmocrOnnx(Model(Jersey.DetModel), Model(Jersey.RecModel), Model(Jersey.Charset)));

        var voteAttributes = new List<string>();
        if (hasOcr) voteAttributes.Add("jersey_number");
        if (hasReid) voteAttributes.Add("role");
        if (voteAttributes.Count > 0)
            modules.Add(new MajorityVoteTracklet(voteAttributes.ToArray()));
        if (RoleCorrection.Enabled)
        {
            if (!hasReid || !hasCalib)
                throw new InvalidOperationException(
                    "role_correction needs reid (role, torso_color) and calibration (bbox_pitch)");
            modules.Add(new TrackletRoleCorrection(RoleCorrection.ToOptions(), log));
        }
        if (hasReid)
        {
            modules.Add(new TrackletTeamClustering(teamFeature));
            if (hasCalib)
                modules.Add(new TrackletTeamSideLabeling());
        }

        log?.LogInformation("reid={Reid} calibration={Calib} ocr={Ocr}", hasReid, hasCalib, hasOcr);
        return new Pipeline(modules, log);
    }
}
