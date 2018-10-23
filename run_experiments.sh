echo "" &&
echo "Running experiments:" &&
echo ".... Running fixed split." &&
cd modeling &&
cd main/fixed_split && python experiment.py && cd ../.. &&
echo "....Running cross-validation." &&
cd main/crossvalidation && python experiment.py && python ttest.py > ttest.txt &&
cd ../.. &&
cd .. && echo "Experiments completed." && echo ""