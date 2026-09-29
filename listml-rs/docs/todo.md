# TODO

## 未実装機能

- `Network`
  - まあここで`Backend`も隠蔽するのがいいかな
  - その場合`Layer`の追加はどうするのか
    - 所有権を受け取るようにする
  - 型レベルではないが，`has_cache`みたいなものを追加するか?
    - それぐらいならwgpuの標準のpanicでいいのでは?

## 演算の整合性

- `gate`の形状確認
  - とりあえず`assert_eq!`
  - 型エラーにしてほしいが現実的でない

- `Backend`と`Tensor`の対応
  - 同じ型なので，そのインスタンスがそのTensorを持っているか静的に検査することは難しい
  - 発生頻度的に低優先度とする

- `forward`前の`backward`
  - networkで`learn_step`として抽象化するとかかな
  - `Layer`を隠蔽することは避けたい
  - 発生頻度的に低優先度とする


## 数式

- `CrossEntropy.loss` の `log` にDELTAを足す
  - Pythonも含めて

## テスト

- `gate`への単体テスト