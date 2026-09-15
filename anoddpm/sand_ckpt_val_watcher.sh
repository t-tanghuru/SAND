#!/bin/bash
# SAND args201 pipeline while training runs on GPU 0 (all automatic, tmux sand-ckpt-val):
#   every checkpoint (except 0 and 3000): validation at lambda 100/250/500 on GPUs 1-2 -> checkpoint curves
#   checkpoint 3000 (300k): validation lambda curve (8 lambdas) -> pick lambda* (best subject-level hippocampus AUROC,
#                           ties -> smaller lambda) -> test the 300k model once at lambda* (interim, for the abstract draft)
#   training finished (500k, params-final.pt): test once at the same pre-chosen lambda* on GPUs 0-2 (final result)
cd /home/tjdnjs/LimLab/AnoDDPM
export PATH=/scratch/users/tjdnjs/conda-envs/anoddpm/bin:$PATH PYTHONNOUSERSITE=1
SPLIT=/home/tjdnjs/LimLab/SAND/data/splits/split_3T_cn1102_ad328.csv
OUT=./sand_eval/ckpt_curve_201
TEST=./sand_eval/test_201
CK=./model/diff-params-ARGS=201/checkpoint
LAMBDA_GRID="50 100 150 200 250 300 400 500"
mkdir -p $OUT $TEST

run_sharded() {  # run_sharded "GPU list" name eval-args...
  local gpus=($1); local name=$2; shift 2; local n=${#gpus[@]}; local ok=1
  for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} python -u sand_eval_ad.py 201 "$@" --split_csv $SPLIT --shard $i/$n > logs/${name}_shard$i.log 2>&1 &
  done
  wait
  for i in $(seq 0 $((n - 1))); do grep -q "done in" logs/${name}_shard$i.log || ok=0; done
  return $((1 - ok))
}

pick_lambda() {
  python - "$@" <<'PY'
import sys, pandas as pd
from sklearn.metrics import roc_auc_score
d = pd.concat(pd.read_csv(p) for p in sys.argv[1:]).drop_duplicates(["tag", "lam"])
auc = {lam: roc_auc_score(g.label, g.hippo_mse) for lam, g in d.groupby("lam")}
best = max(sorted(auc), key=lambda l: (round(auc[l], 6), -l))
print(best, " ".join(f"{l}:{a:.4f}" for l, a in sorted(auc.items())))
PY
}

while true; do
  # the final test has priority over any checkpoint validations still queued
  if grep -q TRAIN_EXIT logs/train_args201_simplex_3T.log && [ -e "$OUT/chosen_lambda.txt" ] && [ ! -e "$TEST/done_final" ]; then
    LAM=$(cat $OUT/chosen_lambda.txt)
    echo "$(date +%H:%M) final 500k test at lambda=$LAM"
    if run_sharded "0 1 2" test_201_500k --split test --lambdas $LAM --ckpt params-final.pt --out_dir $TEST; then
      touch "$TEST/done_final"
      python sand_analyze.py --meta_csv $SPLIT $TEST/args201_test_shard*_subjects.csv > $TEST/analysis_500k.txt 2>&1
      echo "$(date +%H:%M) TEST_500K_DONE"; cat $TEST/analysis_500k.txt | grep -v Warn
    else echo "$(date +%H:%M) TEST_500K_FAILED"; tail -3 logs/test_201_500k_shard*.log; sleep 600; fi
  fi
  for c in $(ls $CK/diff_epoch=*.pt 2>/dev/null | sort -t= -k2 -n); do
    e=$(basename $c .pt | cut -d= -f2)
    grep -q TRAIN_EXIT logs/train_args201_simplex_3T.log && [ -e "$OUT/chosen_lambda.txt" ] && [ ! -e "$TEST/done_final" ] && break
    [ "$e" -eq 0 ] && continue
    [ -e "$OUT/done_$e" ] && continue
    [ $(( $(date +%s) - $(stat -c %Y $c) )) -lt 120 ] && continue
    if [ "$e" -eq 3000 ]; then
      echo "$(date +%H:%M) lambda curve on 300k checkpoint: $LAMBDA_GRID"
      if run_sharded "1 2" lamcurve_201_e3000 --split val --lambdas $LAMBDA_GRID --ckpt checkpoint/diff_epoch=3000.pt --out_dir $OUT; then
        touch "$OUT/done_3000"
        files=$(ls $OUT/args201_val_checkpoint_diff_epoch=3000_shard*_subjects.csv)
        python sand_plot_curves.py lambda $OUT/lambda_curve_300k.png $files > logs/lamcurve_201_plot.log 2>&1
        read LAM SCORES <<< "$(pick_lambda $files)"
        echo "$LAM" > $OUT/chosen_lambda.txt
        echo "$(date +%H:%M) LAMBDA_CURVE_DONE chosen lambda=$LAM | hippo AUROC by lambda $SCORES"
        if run_sharded "1 2" test_201_300k --split test --lambdas $LAM --ckpt checkpoint/diff_epoch=3000.pt --out_dir $TEST; then
          python sand_analyze.py --meta_csv $SPLIT $TEST/args201_test_checkpoint_diff_epoch=3000_shard*_subjects.csv > $TEST/analysis_300k.txt 2>&1
          echo "$(date +%H:%M) TEST_300K_DONE"; cat $TEST/analysis_300k.txt | grep -v Warn
        else echo "$(date +%H:%M) TEST_300K_FAILED"; tail -3 logs/test_201_300k_shard*.log; fi
      else echo "$(date +%H:%M) LAMBDA_CURVE_FAILED"; tail -3 logs/lamcurve_201_e3000_shard*.log; sleep 600; fi
    else
      echo "$(date +%H:%M) validating epoch $e"
      if run_sharded "1 2" ckptval_201_e$e --split val --lambdas 100 250 500 --ckpt checkpoint/diff_epoch=$e.pt --out_dir $OUT; then
        touch "$OUT/done_$e"
        python sand_plot_curves.py ckpt $OUT/ckpt_curve.png $OUT/*_subjects.csv > logs/ckptval_201_curve.log 2>&1
        echo "$(date +%H:%M) CKPT_VAL_DONE epoch $e (iterations $((e * 100)))"
      else echo "$(date +%H:%M) CKPT_VAL_FAILED epoch $e"; tail -3 logs/ckptval_201_e${e}_shard*.log; sleep 600; fi
    fi
  done
  if [ -e "$TEST/done_final" ]; then
    pending=$(ls $CK/diff_epoch=*.pt | while read c; do e=$(basename $c .pt | cut -d= -f2); [ $e -ne 0 ] && [ ! -e $OUT/done_$e ] && echo x; done)
    [ -z "$pending" ] && { echo "$(date +%H:%M) ALL_DONE"; break; }
  fi
  sleep 300
done
