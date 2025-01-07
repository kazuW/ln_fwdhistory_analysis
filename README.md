Lightning NetworkのRouting情報をAIに解析させる。

準備
----- LND -----
LNDのforwadin情報をfileを用意する（この情報を解析）。 main.py 238行目(file名修正必要)
 --> Lncliで取得: ex.) lncli fwdinghistory --start_time "-1w" --max_events 1000 >> file名
LNDのchannel情報を用意する。main.py 240行目(file名修正必要)
 --> Lncliで取得: ex.) lncli listpeers >> file名

----- openAI api key -----
1.login or signup
  https://platform.openai.com/docs/overview
2.dashboard tab
  API Keys -> +Create new secret key -> Create secret key -> copyして.envにpaste
3.dashboard tab
  Usage -> Increase limit -> Buy credits から課金する。
  
