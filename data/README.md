# dialog_log

対話破綻のデータを格納するためのディレクトリです。
対話破綻のデータについては、検出チャレンジの公式サイトから手に入れることができます。これらを、train/testに割り振りフォルダ内に格納してください。

[対話破綻検出チャレンジ/開発データ・評価データ](https://sites.google.com/site/dialoguebreakdowndetection/dev_data)

# trained_model

提案モデルの学習済みモデルを格納しています。
なお、`tokened_vector.zip`は学習済みの分散表現です。日本語WikipediaをfastTextで、語彙数30万で学習させたものです。
単語から単語IDへの変換はvocabファイルで、単語IDから分散表現はvecファイルで取得できます。
