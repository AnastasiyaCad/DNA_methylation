# pip install lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score


def lightgbm_train(X_train, y_train, X_test, y_test, X_val, y_val, n_class):
    train_dataset = lgb.Dataset(X_train, y_train)
    test_dataset = lgb.Dataset(X_test, y_test)
    val_dataset = lgb.Dataset(X_val, y_val)
    evals_result = {}
    max_epochs = 100
    booster = lgb.train({"objective": "multiclass", "num_class": n_class, "verbosity": -1},
                        train_set=train_dataset, valid_sets=[train_dataset, val_dataset], num_boost_round=max_epochs,
                        evals_result=evals_result)
    # booster = lgb.train({"objective": "multiclass", "num_class": n_class, "verbosity": -1},
    #                     train_set=train_dataset, valid_sets=(test_dataset,),
    #                     num_boost_round=16)
    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    test_preds = np.argmax(test_preds, axis=1)
    train_preds = np.argmax(train_preds, axis=1)


    loss_info = {
        'epoch': list(range(len(evals_result['training']['multi_logloss']))),
        'train/loss': evals_result['training']['multi_logloss'],
        'val/loss': evals_result['valid_1']['multi_logloss']
    }

    print("\nTest  Accuracy : %.2f" % accuracy_score(y_test, test_preds))
    print("Train Accuracy : %.2f" % accuracy_score(y_train, train_preds))

    return test_preds, train_preds, loss_info


def lightgbm_LGBMClassifier(X_train, y_train, X_test, y_test, X_val, y_val, n_class, max_epochs, batch_size):
    booster = lgb.LGBMClassifier(objective="multiclassova", n_estimators=10, num_class=n_class)
    booster.fit(
        X_train, y_train,
        max_epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    history_loss = booster.history['train']['loss']

    print("\nTest  Accuracy LGBMClassifier : %.2f" % booster.score(X_test, y_test))
    print("Train Accuracy LGBMClassifier : %.2f" % booster.score(X_train, y_train))

    return test_preds, train_preds, proba_test, proba_train, history_loss


