seed=42
python ./medeha/preprocess/mimic_cxr.py --seed "$seed"
python ./medeha/preprocess/chexpert.py --seed "$seed"
python ./medeha/preprocess/rsna.py --seed "$seed"
python ./medeha/preprocess/covidx.py --seed "$seed"
python ./medeha/preprocess/siim.py --seed "$seed"
python ./medeha/preprocess/object_cxr.py --seed "$seed"