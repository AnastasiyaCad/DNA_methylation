import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score


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


def xgboost_train_(X_train, y_train, X_test, y_test, X_val, y_val, feature_names, n_class):
    early_stopping_rounds = 10
    num_boost_round = 16
    max_epochs = 500

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    dtest = xgb.DMatrix(data=X_test)

    model_params = {'max_depth': 5, 'eta': 1, 'objective': 'multi:softmax', 'num_class': n_class}
    #booster = xgb.train({'max_depth': 5, 'eta': 1, 'objective': 'multi:softmax', 'num_class': n_class},
    #                    um_boost_round=num_boost_round,
    #                    dtrain=train_dataset,
    #                    evals=[(train_dataset, "train"), (val_dataset, "val")],
    #                    early_stopping_rounds=early_stopping_rounds)
    evals_result = {}
    booster = xgb.train(
        params=model_params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        num_boost_round=max_epochs,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    test_preds = booster.predict(dtest)
    train_preds = booster.predict(dtrain)
    
    loss_info = {
                'epoch': list(range(len(evals_result['train']['mlogloss']))),
                'train/loss': evals_result['train']['mlogloss'],
                'val/loss': evals_result['val']['mlogloss']
    }

    print("\nTest  Accuracy : %.2f" % accuracy_score(y_test, booster.predict(data=dtest)))
    print("Train Accuracy : %.2f" % accuracy_score(y_train, booster.predict(data=dtrain)))

    return test_preds, train_preds, loss_info


def xgboost_XGBClassifier(X_train, y_train, X_test, y_test, X_val, y_val, n_class, max_epochs, batch_size):
    xgbc = xgb.XGBClassifier(learning_rate=0.5,
                             n_estimators=150,
                             max_depth=6,
                             min_child_weight=0,
                             gamma=0,
                             reg_lambda=1,
                             subsample=1,
                             colsample_bytree=0.75,
                             scale_pos_weight=1,
                             objective='multi:softprob',
                             num_class=n_class,
                             random_state=42)

    mcl = xgbc.fit(
        X_train, y_train, batch_size=batch_size, epochs=max_epochs, validation_data=(X_val, y_val))
    proba_train = mcl.predict_proba(X_train)
    proba_test = mcl.predict_proba(X_test)

    test_preds = mcl.predict(X_test)
    train_preds = mcl.predict_proba(X_test)

    history_loss = mcl.history['train']['loss']

    print("\nTest  Accuracy XGBClassifier : %.2f" % mcl.score(X_test, y_test))
    print("Train Accuracy XGBClassifier : %.2f" % mcl.score(X_train, y_train))

    return test_preds, train_preds, proba_test, proba_train, history_loss








