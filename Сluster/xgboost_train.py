import xgboost as xgb


def xgboost_main(X_train, y_train, X_test, y_test, n_class):
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test)

    num_boost_round = 16
    # param train model
    params = {
        # максимальная глубина дерева по умолч 6
        'max_depth': 3,
        'objective': 'multi:softmax',  # error evaluation for multiclass training
        'num_class': n_class,
        'n_gpus': 0
    }
    booster = xgb.train(params, um_boost_round=num_boost_round, dtrain=dtrain,
                    evals=[(dtrain, "train"), (dtest, "test")])

    y_pred = booster.predict(dtest)

    print("\nTest  Accuracy : %.2f" % booster.score(X_test, y_test))
    print("Train Accuracy : %.2f" % booster.score(X_train, y_train))

    return booster, y_pred


def xgboost_train(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class):
    early_stopping_rounds = 10
    num_boost_round = 16

    train_dataset = xgb.Dataset(X_train, y_train, feature_name=feature_names)
    test_dataset = xgb.Dataset(X_test, y_test, feature_name=feature_names)
    val_dataset = xgb.Dataset(X_val, y_val, feature_name=feature_names)

    booster = xgb.train({'max_depth': 5, 'eta': 1, 'objective': 'multi:softmax', 'num_class': n_class},
                        um_boost_round=num_boost_round,
                        dtrain=train_dataset,
                        evals=[(train_dataset, "train"), (val_dataset, "val")],
                        early_stopping_rounds=early_stopping_rounds)
    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    print("\nTest  Accuracy : %.2f" % booster.score(X_test, y_test))
    print("Train Accuracy : %.2f" % booster.score(X_train, y_train))

    return test_preds, train_preds








