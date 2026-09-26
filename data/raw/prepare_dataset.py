"""
prepare_dataset.py

Run this from inside the folder containing your downloaded ImageNet
{wnid}.tar files (e.g. data/raw/).

What it does:
1. Maps known wnids -> human-readable class names (for the PRISM paper replication)
2. For each tar file present, extracts only the first N images (default 5)
3. Saves them into subfolders named after the class (e.g. data/raw/timber_wolf/)
4. Handles both flat and nested tar structures automatically
5. Prints a summary table at the end
6. Optionally deletes the original tar files after successful extraction (off by default)

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --n_images 8
    python prepare_dataset.py --delete_tars
"""

import os
import tarfile
import argparse
import shutil

# -----------------------------------------------------------------
# Map ImageNet WordNet IDs -> friendly class names used in the
# PRISM paper replication (Szandała, 2023)
# -----------------------------------------------------------------
WNID_TO_NAME = {
    "n02114548": "timber_wolf",
    "n02114855": "coyote",
    "n02120505": "grey_fox",
    "n02114367": "red_wolf",
    "n02106166": "border_collie",
    "n02110185": "siberian_husky",
    "n02111889": "samoyed",
    "n03272562": "electric_locomotive",
    "n03393912": "passenger_car",
    "n04310018": "steam_locomotive",
    "n07697313": "cheeseburger",
    "n07693725": "bagel",
    "n01774384": "black_widow",
    "n01773797": "garden_spider",
}

VALID_EXTENSIONS = (".jpeg", ".jpg", ".JPEG", ".JPG")


def extract_images_from_tar(tar_path, out_dir, n_images):
    """
    Extract up to n_images JPEG files from a tar archive into out_dir,
    flattening any nested folder structure inside the tar.
    Returns the number of images successfully extracted.
    """
    os.makedirs(out_dir, exist_ok=True)
    extracted = 0

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        image_members = [m for m in members if m.name.endswith(VALID_EXTENSIONS)]

        if not image_members:
            print(f"  [WARNING] No JPEG members found directly in {tar_path}. "
                  f"First few entries were: {[m.name for m in members[:5]]}")
            return 0

        for m in image_members[:n_images]:
            # Flatten path: some tars nest images inside a wnid-named subfolder
            fname = os.path.basename(m.name)
            target_path = os.path.join(out_dir, fname)

            fileobj = tar.extractfile(m)
            if fileobj is None:
                continue

            with open(target_path, "wb") as f_out:
                f_out.write(fileobj.read())

            extracted += 1

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Prepare PRISM replication dataset from ImageNet tar files.")
    parser.add_argument("--n_images", type=int, default=5,
                         help="Number of images to extract per class (default: 5)")
    parser.add_argument("--delete_tars", action="store_true",
                         help="Delete tar files after successful extraction to save disk space")
    parser.add_argument("--source_dir", type=str, default=".",
                         help="Folder containing the .tar files (default: current directory)")
    args = parser.parse_args()

    source_dir = args.source_dir
    summary = []

    print(f"Scanning '{source_dir}' for known class tar files...\n")

    for wnid, class_name in WNID_TO_NAME.items():
        tar_filename = f"{wnid}.tar"
        tar_path = os.path.join(source_dir, tar_filename)

        if not os.path.exists(tar_path):
            print(f"[SKIP] {tar_filename} not found (class: {class_name})")
            summary.append((class_name, wnid, 0, "tar not found"))
            continue

        out_dir = os.path.join(source_dir, class_name)
        print(f"[PROCESSING] {tar_filename} -> {class_name}/")

        try:
            count = extract_images_from_tar(tar_path, out_dir, args.n_images)
            status = "OK" if count > 0 else "FAILED (0 images extracted)"
            summary.append((class_name, wnid, count, status))
            print(f"  -> extracted {count} images\n")

            if args.delete_tars and count > 0:
                os.remove(tar_path)
                print(f"  -> deleted {tar_filename} to save space\n")

        except tarfile.ReadError as e:
            print(f"  [ERROR] Could not read {tar_path}: {e}\n")
            summary.append((class_name, wnid, 0, f"read error: {e}"))

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Class':<22}{'WNID':<14}{'Images':<10}{'Status'}")
    print("-" * 60)
    for class_name, wnid, count, status in summary:
        print(f"{class_name:<22}{wnid:<14}{count:<10}{status}")
    print("=" * 60)

    total_images = sum(c for _, _, c, _ in summary)
    total_classes_ok = sum(1 for _, _, c, _ in summary if c > 0)
    print(f"\nTotal: {total_images} images across {total_classes_ok}/{len(WNID_TO_NAME)} classes.")

    if any(c == 0 for _, _, c, _ in summary):
        print("\nSome classes failed or were missing. Check messages above.")
        print("If extraction failed with 0 images despite the tar existing, "
              "run the diagnostic snippet mentioned in chat to inspect tar internal structure.")


if __name__ == "__main__":
    main()