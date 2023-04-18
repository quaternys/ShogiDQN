# DQNShogi
- USIプロトコル対応の将棋AI．
- DQNでゼロから学習しています（手法の詳細は省略）．


# 使い方
- まずPyTorch環境を用意し，`main.bat` 内の実行コマンドをその環境の `python.exe` に置き換えます．
  - GPU/CPU どちらでも動きます．
- [ShogiGUI](http://shogigui.siganus.com/)や[将棋所](http://shogidokoro.starfree.jp/)から使う場合，`main.bat` をエンジンとして登録します．
  - GUIの操作方法等はダウンロード元を参照してください．
- あとは対局タブとかから遊んでください．

## パラメータ
- DepthLimit (1-8): 探索深度．
  - ※探索手法は minimax探索＋最良優先探索＋αβ枝刈り＋5手前向き枝刈り
- Temperature (0-2): Boltzmann方策の温度．
  - 0なら最善手．温度が高いほどランダム性が増します．
  - 探索終了後に合法手の評価値に対してBoltzmann方策で手を決定します．
  - 学習時は1で統一しています．


