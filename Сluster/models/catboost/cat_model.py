import catboost
import pandas as pd
from catboost import CatBoostClassifier, Pool, metrics

import globalConstants


def CatBoostTrainModel(X_train, y_train, X_val, y_val, X_test, y_test, path=globalConstants.fNameSaveOutput):
    train_pool = Pool(X_train, y_train)
    validate_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=globalConstants.EPOCHS,
        learning_rate=1,
        #n_estimators=globalConstants.NUM_CLASSES,
        #eval_metric=metrics.Accuracy(),
        verbose=False,
        loss_function='MultiClass'
    )
    model.fit(train_pool, eval_set=validate_pool)
    test_preds = model.predict(X_test)
    test_preds = [x for x, in test_preds]
    train_preds = model.predict(X_train)
    test_preds = [x for x, in train_preds]
    train_preds_proba = model.predict_proba(X_train)
    val_preds_proba = model.predict_proba(X_val)
    test_preds_proba = model.predict_proba(X_test)

    metrics_train = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
    metrics_val = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")

    loss_info = {
        'epoch': range(globalConstants.EPOCHS),
         'train/loss': model.eval_metrics(train_pool, [metrics.MultiClass()])['MultiClass'],
         'val/loss': model.eval_metrics(validate_pool, [metrics.MultiClass()])['MultiClass']
    }

    return train_preds_proba, val_preds_proba, test_preds_proba, loss_info


def cat(X_train, y_train, X_val, y_val, X_test, y_test):
    train_pool = Pool(X_train, y_train)
    validate_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        n_estimators=10,
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