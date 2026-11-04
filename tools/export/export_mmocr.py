"""Export the mmocr text detector (DBNet) and recognizer (SAR) to ONNX for MmocrOnnx.

Not used for the published models: those are PP-OCRv4, see export_ppocr.sh. This is the route
to the original baseline's weights; mmdeploy needs Python <= 3.11 and torch <= 2.1.

The baseline uses det='dbnet_resnet18_fpnc_1200e_icdar2015' and rec='SAR'.
SAR has an attention decoder, which is easiest to export via mmdeploy:

    pip install mmdeploy==1.3.1
    python tools/deploy.py configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py \
        <mmocr>/configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py \
        dbnet_resnet18_fpnc_1200e_icdar2015_*.pth demo_text_det.jpg --work-dir out_det
    python tools/deploy.py configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py \
        <mmocr>/configs/textrecog/sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py \
        sar_*.pth demo_text_recog.jpg --work-dir out_rec

Then copy:
  out_det/end2end.onnx -> pretrained_models/mmocr_det.onnx   (output: prob map [N,1,H,W])
  out_rec/end2end.onnx -> pretrained_models/mmocr_rec.onnx   (output: logits [N,T,C])
and dump the recognizer dictionary (one char per line) with this script:

    python export_mmocr.py dump-charset

MmocrOnnx uses RecDecodeMode.Attention for SAR (stop at <EOS> = first class beyond
the charset). If you swap the recognizer for a CTC model (CRNN/SVTR), pass
RecDecodeMode.Ctc instead (blank = last class).
"""
import sys

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "dump-charset":
    from mmocr.apis import TextRecInferencer

    inferencer = TextRecInferencer("SAR", device="cpu")
    dictionary = inferencer.model.decoder.dictionary
    with open("mmocr_charset.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dictionary.dict))
    print(f"wrote mmocr_charset.txt ({dictionary.num_classes} classes)")
