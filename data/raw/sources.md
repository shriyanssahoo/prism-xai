# Dataset Sources

Images were obtained from the official ImageNet website (image-net.org),
downloaded as per-class tar archives using the URL pattern:
https://image-net.org/data/winter21_whole/{wnid}.tar

WordNet IDs (wnids) used for this replication:

| Class | WNID |
|---|---|
| timber_wolf | n02114548 |
| coyote | n02114855 |
| grey_fox | n02120505 |
| red_wolf | n02114367 |
| border_collie | n02106166 |
| siberian_husky | n02110185 |
| samoyed | n02111889 |
| electric_locomotive | n03272562 |
| passenger_car | n03393912 |
| steam_locomotive | n04310018 |
| garden_spider | n01773797 |

10 images were extracted per class using `data/raw/prepare_dataset.py`.
Requires a registered ImageNet account to access the download.