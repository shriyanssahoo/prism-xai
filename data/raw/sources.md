# Dataset Sources

Images were obtained from the official ImageNet website (image-net.org), downloaded as per-class tar archives using the URL pattern:

```
https://image-net.org/data/winter21_whole/{wnid}.tar
```

Access requires a registered, approved ImageNet account.

## WordNet IDs (wnids) Used in This Replication

| Class               | WNID      |
| ------------------- | --------- |
| timber_wolf         | n02114548 |
| coyote              | n02114855 |
| grey_fox            | n02120505 |
| red_wolf            | n02114367 |
| border_collie       | n02106166 |
| siberian_husky      | n02110185 |
| samoyed             | n02111889 |
| electric_locomotive | n03272562 |
| passenger_car       | n03393912 |
| steam_locomotive    | n04310018 |
| cheeseburger        | n07697313 |
| bagel               | n07693725 |
| black_widow         | n01774384 |
| garden_spider       | n01773797 |

These classes were chosen to match the example classes used in the original PRISM paper (Szandała, 2023), specifically the wolf/coyote/grey-fox comparison (Section 3.3, Fig. 8), the canine confusion set (Section 5, Fig. 13), and the train/food comparisons (Section 3.1, Figs. 3 and 5).

## Extraction Process

10 images were extracted per class using the script `data/raw/prepare_dataset.py`, which:

1. Scans the working directory for `{wnid}.tar` files
2. Extracts the first N JPEG images per archive (default N=10) into a class-named subfolder
3. Flattens any nested folder structure inside the tar
4. Optionally deletes the source tar files afterward to save disk space

Usage:

```bash
cd data/raw
python prepare_dataset.py --n_images 10
```

## License / Usage Note

ImageNet images are provided for non-commercial research and educational use only, per ImageNet's terms of access. This dataset subset is used here strictly for a course replication assignment.
