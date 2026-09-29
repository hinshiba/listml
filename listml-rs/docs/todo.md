# TODO

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

## ソフトウェア設計

- `gate`が`Rectfier`にある
  - 分離する

## テスト

- `gate`への単体テスト