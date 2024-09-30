CluStream(https://www.vldb.org/conf/2003/papers/S04P02.pdf)の簡易的なjava実装です

Micro-clusterの初期化、要素の追加についてのみ実装を行い、スナップショットの作成の実装は行っていません

streamデータとしてTweet classを定義しており、これはtweetの内容(value)と作成時間(timestamp)を持っています. tweetデータはOpenAI Embedding等で作成した埋め込みベクトルデータを想定しています
