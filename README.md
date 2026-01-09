Training:
Energy only:
python main.py train  --xyz train.xyz  --out ace_linear_model.pt  --device cpu  --dtype float64  --rc 3.0 --lmax 2 --nmax 4  --lam 5.0  --train_frac 0.8  --ridge_lambda 1e-8

Energy+Force:
python main.py train --xyz train_try_qe.xyz --out mgcoh.pt --device cuda --dtype float64  --rc 6.0 --lmax 3 --nmax 10 --lam 5.0  --train_frac 0.8 --use_forces --epochs 200 --lr 0.05 --w_energy 1.0  --w_forces 100.0


Evaluation:
Energy:
python main.py eval --xyz test.xyz --model ace_linear_model.pt --device cpu  --dtype float64

Energy+Forces:
python main.py eval --xyz mgo.xyz --model mgcoh.pt --eval_forces --device cpu --dtype float64

Prediction:

python main.py predict --xyz input.xyz --model model.pt --out pred.xyz --device cpu --dtype float64