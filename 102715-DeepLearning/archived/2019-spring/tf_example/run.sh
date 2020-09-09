#/bin/bash
#BSUB -J JOBNAME
#BSUB -e LOG_DIR/ERROR_FILENAME_%J.err
#BSUB -o LOG_DIR/OUTPUT_FILENAME_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=1]"
python CODE_DIR/CODE_FILENAME.py