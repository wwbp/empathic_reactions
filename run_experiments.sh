echo "" &&
echo "Running experiments:" &&
cd modeling &&
echo "....Running cross-validation." &&
cd main/crossvalidation && python experiment.py && python ttest.py > ttest.txt &&
cd ../.. &&
cd .. && echo "Experiments completed." && echo ""