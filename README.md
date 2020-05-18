# reliability_diagrams
以下で実装されているReliability diagramの描画スクリプトを少し書き換えたもの。
https://github.com/mbernste/reliability-diagrams
pureにfor loopで処理されていた部分をnumpyの関数に置き換えた。
デフォルトの設定でlen(predictions) == 10000なら1.0sで結果が出る。

Bröcker, J. and Smith, L. (2007). Increasing the reliability of reliability diagrams. Weather and Forecasting,22(3), 651–661
