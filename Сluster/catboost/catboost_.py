# pip install catboost
import catboost
from catboost import CatBoost
from catboost import CatBoostClassifier
from catboost.utils import eval_metric
from catboost import CatBoostClassifier, Pool, metrics, cv
import numpy as np


# def catboost_CatBoost(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class, iterations, max_epochs,
#                       batch_size):
#     early_stopping_rounds = 10
#     booster = CatBoost(params={'iterations': iterations, 'verbose': 10, 'loss_function': 'MultiClass',
#                                'classes_count': n_class})
#     #booster.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
#     booster.fit(
#         X_train, y_train
#     )
#     booster.set_feature_names(feature_names)
#
#     test_preds = booster.predict(X_test, prediction_type="Class").flatten()
#     train_preds = booster.predict(X_train, prediction_type="Class").flatten()
#
#     history_loss = booster.history['train']['loss']
#
#     print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "Accuracy")[0])
#     print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "Accuracy")[0])
#
#     print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "F1")[0])
#     print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "F1")[0])
#
#     return test_preds, train_preds, history_loss
#
#
# def catboost_CatBoostClassifier(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, iterations, max_epochs,
#                                 batch_size):
#     early_stopping_rounds = 10
#     epophs = 500
#     train_dataset = cb.Pool(X_train, y_train)
#     val_dataset = cb.Pool(X_val, y_val)
#     test_dataset = cb.Pool(X_test, y_test)
#
#     booster = cb.CatBoostClassifier(
#         learning_rate=0.055,
#         n_estimators=epophs,
#         random_seed=43,
#         loss_function='MultiClass'
#     )
#
#     model_params = {'max_depth': 5, 'eta': 1, 'objective': 'multi:softmax', 'num_class': 25}
#     evals_result = {}
#
#     booster.train(
#         params=model_params,
#         dtrain=train_dataset,
#         evals=[(train_dataset, "train"), (val_dataset, "val")],
#         num_boost_round=max_epochs,
#         early_stopping_rounds=early_stopping_rounds,
#         evals_result=evals_result,
#         verbose_eval=False
#     )
#
#     test_preds = booster.predict(X_test)
#     train_preds = booster.predict(X_train)
#
#     print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "Accuracy")[0])
#     print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "Accuracy")[0])
#
#     print("\nTest  Accuracy : %.2f"%eval_metric(y_test, test_preds, "F1")[0])
#     print("Train Accuracy : %.2f"%eval_metric(y_train, train_preds, "F1")[0])
#
#     return test_preds, train_preds, proba_test, proba_train, history_loss


def cat(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, iterations, max_epochs, batch_size, categorical_features_indices):
    train_pool = Pool(X_train, y_train)
    validate_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        n_estimators=10,
        #iterations=100,
        eval_metric=metrics.Accuracy(),
        random_seed=42,
        verbose=False,
        loss_function='MultiClass'
    )

    model.fit(train_pool, eval_set=validate_pool)

    test_preds = model.predict(X_test)
    train_preds = model.predict(X_train)

    loss_info = {
        'epoch': list(range(len(model.eval_metrics(train_pool, [metrics.MultiClass()])['MultiClass']))),
        'train/loss': model.eval_metrics(train_pool, [metrics.MultiClass()])['MultiClass'],
        'val/loss': model.eval_metrics(validate_pool, [metrics.MultiClass()])['MultiClass']
    }

    return test_preds, train_preds, loss_info











