# pip install catboost
from catboost import CatBoostClassifier
from catboost import CatBoost
from catboost.utils import eval_metric


def catboost_CatBoost(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, iterations):
    early_stopping_rounds = 10
    booster = CatBoost(params={'iterations': iterations, 'verbose': 10, 'loss_function': 'MultiClass',
                               'classes_count': n_class})
    booster.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
    booster.set_feature_names(feature_names)

    test_preds = booster.predict(X_test, prediction_type="Class").flatten()
    train_preds = booster.predict(X_train, prediction_type="Class").flatten()

    print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "Accuracy")[0])
    print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "Accuracy")[0])
    
    print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "F1")[0])
    print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "F1")[0])

    return test_preds, train_preds


def catboost_CatBoostClassifier(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, iterations):
    early_stopping_rounds = 10
    booster = CatBoostClassifier(
        iterations=iterations,
        random_seed=43,
        loss_function='MultiClass'
    )
    booster.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
    booster.set_feature_names(feature_names)
    # booster.fit(X_train, y_train, eval_set=(X_test, y_test))
    # booster.fit(
    #     X_train, y_train,
    #     cat_features=feature_names,
    #     verbose=50
    # )

    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "Accuracy")[0])
    print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "Accuracy")[0])
    
    print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "F1")[0])
    print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "F1")[0])

    return test_preds, train_preds


