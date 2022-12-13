import globalConstants
from loadCreateData.CreateTrainTestData import getDataTrainValFromCrossValTest
import models.xgboost.xgb_model as xgbmodel


def main():
    XTrainCrossVal, yTrainCrossVal, XValCrossVal, yValCrossVal, DataBetaNamePersonTrainSet, DataBetaNamePersonTestSet = getDataTrainValFromCrossValTest()
    xgbmodel.getTablesGraphXGBoost(XTrainCrossVal, yTrainCrossVal, XValCrossVal, yValCrossVal, DataBetaNamePersonTrainSet, DataBetaNamePersonTestSet)


if __name__ == "__main__":
    main()
