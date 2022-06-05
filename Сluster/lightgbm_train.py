# pip install lightgbm
import lightgbm as lgb


def lightgbm_train(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class):
    train_dataset = lgb.Dataset(X_train, y_train, feature_name=feature_names)
    test_dataset = lgb.Dataset(X_test, y_test, feature_name=feature_names)
    val_dataset = lgb.Dataset(X_val, y_val, feature_name=feature_names)

    booster = lgb.train({"objective": "multiclass", "num_class": n_class, "verbosity": -1},
                        train_set=train_dataset, valid_sets=(val_dataset,),
                        num_boost_round=10)
    # booster = lgb.train({"objective": "multiclass", "num_class": n_class, "verbosity": -1},
    #                     train_set=train_dataset, valid_sets=(test_dataset,),
    #                     num_boost_round=16)
    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    print("\nTest  Accuracy : %.2f" % booster.score(X_test, y_test))
    print("Train Accuracy : %.2f" % booster.score(X_train, y_train))

    return test_preds, train_preds


def lightgbm_LGBMClassifier(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class):
    booster = lgb.LGBMClassifier(objective="multiclassova", n_estimators=10, num_class=n_class)
    booster.fit(X_train, y_train, eval_set=[(X_val, y_val), ])

    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    print("\nTest  Accuracy : %.2f" % booster.score(X_test, y_test))
    print("Train Accuracy : %.2f" % booster.score(X_train, y_train))

    return test_preds, train_preds


