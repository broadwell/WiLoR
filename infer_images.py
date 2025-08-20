import argparse
import glob
import logging
import os

import cv2
import jsonlines
import numpy as np
import torch
from rich.logging import RichHandler
from ultralytics import YOLO

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import Renderer, cam_crop_to_full


def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2.0, img_res[1] / 2.0]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]

    V_2d = (K @ points.T).T
    return V_2d[..., :-1]


def main():
    parser = argparse.ArgumentParser(description="WiLoR demo code")
    parser.add_argument(
        "--images_path",
        type=str,
        help="Folder with images for hand detection",
    )
    parser.add_argument(
      "--out_folder",
      type=str,
      default="out_demo",
      help="Output folder to save rendered results"
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="If set, overwrite existing output file. If not, append new results.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Set verbose logging.",
    )
    parser.add_argument(
        "--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox"
    )
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument(
        "--file_type",
        nargs="+",
        default=["*.jpg", "*.png", "*.jpeg"],
        help="List of file extensions to consider",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    # Download and load checkpoints
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    detector = YOLO("./pretrained_models/detector.pt")
    # Setup the renderer (disabled)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output render directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    output_path = f"detected_hands.WiLoR.jsonl"
    print("OUTPUT FILE", output_path)

    # Get filename from path img_path
    #img_fn, _ = os.path.splitext(os.path.basename(args.images_path))

    # Iterate over all images in folder
    with jsonlines.open(output_path, mode="w") as writer:
        print(glob.glob(args.images_path + "/*." + args.file_type[0]))
        for fp in glob.glob(args.images_path + "/*." + args.file_type[0]):
            img_fn = os.path.basename(fp)
            print("Processing", fp, img_fn)
            #img_cv2 = image_from_video_frame(str(args.video_path), frameno)
            img_cv2 = cv2.imread(fp)

            detections = detector(img_cv2, conf=0.3, verbose=False)[0]
            print("total detections:", len(detections))
            bboxes = []
            confidence_values = []
            is_right = []
            for det in detections:
                # Some attributes of the detection are only accessible if
                # exported this way (???)
                det_df = det.to_df()
                confidence_values.append(det_df["confidence"].item())

                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            if len(bboxes) == 0:
                continue
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(
                model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=0
            )

            # all_verts_3d = []  # 3D vertices (778)
            # all_cam_t = []  # Camera translation(s) in 3 dimensions
            # all_right = []  # Right = 1, left = 0
            # all_kpts_3d = []  # 3D keypoints (OpenPose 21)
            # all_verts_2d = []  # 2D vertices (778)
            # all_kpts_2d = []  # 2D keypoints (aka joints) (OpenPose 21)

            for batch in dataloader:
                batch = recursive_to(batch, device)

                with torch.no_grad():
                    out = model(batch)

                multiplier = 2 * batch["right"] - 1
                pred_cam = out["pred_cam"]
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = (
                    model_cfg.EXTRA.FOCAL_LENGTH
                    / model_cfg.MODEL.IMAGE_SIZE
                    * img_size.max()
                )
                pred_cam_t_full = (
                    cam_crop_to_full(
                        pred_cam, box_center, box_size, img_size, scaled_focal_length
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # verts = vertices = 778 mesh nodes
                    # joints = keypoints = 21 knuckles, etc. from OpenPose

                    personid = batch["personid"].cpu()[n].item()

                    bbox = bboxes[n]

                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()

                    is_right = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                    cam_t = pred_cam_t_full[n]
                    verts_2d = project_full_img(
                       verts, cam_t, scaled_focal_length, img_size[n]
                    )
                    kpts_2d = project_full_img(
                        joints, cam_t, scaled_focal_length, img_size[n]
                    )
                    global_orient = (
                        out["pred_mano_params"]["global_orient"].cpu().numpy()[n][0],
                    )

                    output_json = [
                        {
                            "image": fp,
                            "personid": personid,
                            "bbox": bbox,
                            "confidence": confidence_values[n],
                            "right": int(is_right),
                            "pred_cam": pred_cam.cpu().numpy().tolist()[0],
                            "cam_t": cam_t.tolist(),
                            # "verts_3d": verts.tolist(),
                            # "verts_2d": verts_2d.cpu().numpy().tolist(),
                            "kpts_3d": joints.tolist(),
                            "kpts_2d": kpts_2d.cpu().numpy().tolist(),
                            "global_orient": np.array(global_orient).tolist()[0],
                        }
                    ]
                    writer.write_all(output_json)

                    # all_verts_3d.append(verts)
                    # all_cam_t.append(cam_t)
                    # all_right.append(is_right)
                    # all_kpts_3d.append(joints)
                    # all_verts_2d.append(verts_2d)
                    # all_kpts_2d.append(kpts_2d)

                    # print(img_fn, n)
                    # print("VERTS:", verts)
                    # print("CAM:", cam_t)
                    # print("RIGHT:", is_right)
                    # print("JOINTS/KPTS:", joints)

                    # Save all meshes to disk
                    LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)
                    if args.save_mesh:
                       camera_translation = cam_t.copy()
                       tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                       tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))


if __name__ == "__main__":
    main()
