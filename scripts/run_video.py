import argparse
import cv2
from PIL import Image
import torch
from dinov2.data.transforms import make_classification_eval_transform


def load_model(model_name: str, device: str = "cpu", local: bool = False, weights: str = None):
    repo = "." if local else "facebookresearch/dinov2"
    source = "local" if local else "github"
    model = torch.hub.load(repo, model_name, source=source)
    if weights is not None:
        state_dict = torch.load(weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run DINOv2 on an mp4 file")
    parser.add_argument("video", help="Path to input mp4 file")
    parser.add_argument(
        "--model",
        default="dinov2_vitb14",
        help="Model name from dinov2.hub (e.g. dinov2_vitb14)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame",
    )
    parser.add_argument(
        "--output",
        default="features.pt",
        help="Where to save the extracted features",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Load model from local repository",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional local path to model weights",
    )
    args = parser.parse_args()

    model = load_model(args.model, device=args.device, local=args.local, weights=args.weights)
    transform = make_classification_eval_transform()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    feats = []
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % args.frame_step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img).unsqueeze(0).to(args.device)
            with torch.no_grad():
                feat = model(tensor)
            feats.append(feat.cpu())
        frame_idx += 1

    cap.release()

    if feats:
        features = torch.cat(feats)
        torch.save(features, args.output)
        print(f"Saved features for {features.shape[0]} frames to {args.output}")
    else:
        print("No frames processed")


if __name__ == "__main__":
    main()
