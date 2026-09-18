| Dataset   | Model         | Params                             |   Accuracy |   Precision |   Recall |   F1 Score |
|:----------|:--------------|:-----------------------------------|-----------:|------------:|---------:|-----------:|
| sephora   | Dummy         |                                    |   0.838693 |    0.838693 | 1        |   0.912271 |
| sephora   | Random Forest | n_estimators=87, max_depth=12      |   0.855034 |    0.865962 | 0.97863  |   0.918855 |
| sephora   | SVM           | C=85.68869785189007, kernel=linear |   0.844491 |    0.845048 | 0.997486 |   0.914961 |
| movies    | Dummy         |                                    |   0.499766 |    0        | 0        |   0        |s
| movies    | Random Forest | n_estimators=51, max_depth=10      |   0.595874 |    0.568654 | 0.795689 |   0.663281 |
| movies    | SVM           | C=0.01129951608310662, kernel=rbf  |   0.499766 |    0        | 0        |   0        |