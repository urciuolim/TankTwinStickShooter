#!/bin/bash
#
#SBATCH --job-name=train_tankai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlu216@nyu.edu

source ~/.bashrc
module purge
module load cuda/11.0.194
conda activate /scratch/mlu216/cenv/TankAI

PY_BASE=~/PythonScripts
GAME_BASE=~/game_server
ENVS=~/envs/7

PREAMBLE=$PY_BASE/StableBaselines_TankEnv_PBT_Preamble.py
GAME=$GAME_BASE/game.x86_64
CONFIG=./Assets/config.json
POP_TRAIN=$PY_BASE/StableBaselines_TankEnv_MatchmakingTraining.py
INDV_TRAIN=$PY_BASE/StableBaselines_TankEnv_IterativeTraining.py
TOURN=$PY_BASE/StableBaselines_TankEnv_Tournament.py
EVAL=$PY_BASE/StableBaselines_TankEnv_Evaluate.py
REPLACE=$PY_BASE/StableBaselines_TankEnv_ExploitExplore.py
BASE_DIR=~/models_pbt/
POP_FILE=~/models_pbt/population.txt
NOUNS=$PY_BASE/nouns.txt
ADJS=$PY_BASE/adjs.txt
GAMELOG="--gamelog ~/models_pbt/gamelog"
CONSOL=$PY_BASE/consolidate.py

EPOCHS=1
PARTS=4
STEPS=10000
TRIALS=50
MIN_STEP=5000

echo "Starting preamble"
cd $ENVS-1/
python $PREAMBLE $GAME $CONFIG $POP_TRAIN $INDV_TRAIN $TOURN $EVAL $REPLACE $BASE_DIR $POP_FILE $NOUNS $ADJS --start 8 $GAMELOG.txt
echo "Preamble complete"

for (( i=1; i<=$EPOCHS; i++ ))
do
	echo "PBT Epoch: $i"
	echo "Starting training"
	date
	for (( j=1; j<=$PARTS; j++ ))
	do
		cd $ENVS-$j/
		python $POP_TRAIN $GAME $CONFIG $INDV_TRAIN $BASE_DIR $POP_FILE \
			--steps $STEPS --rs $GAMELOG.txt --idx $j --part $PARTS &
	done
	wait

	echo "Training complete, starting tournament"
	date
	for (( j=1; j<=$PARTS; j++ ))
	do
		cd $ENVS-$j/
		python $TOURN $GAME $CONFIG $EVAL $BASE_DIR $POP_FILE \
			--num_trials $TRIALS $GAMELOG.txt --idx $j --part $PARTS &
	done
	wait

	echo "Tournament complete, consolidating"
	python $CONSOL $BASE_DIR $POP_FILE $PARTS

	echo "Consolidation complete, considering replacements"
	date
	cd $ENVS-1/
	python $REPLACE $GAME $CONFIG $EVAL $BASE_DIR $POP_FILE $NOUNS $ADJS \
		--num_trials $TRIALS --min_step $MIN_STEP $GAMELOG.txt
	echo "Replacements complete, epoch complete"
	date
done
