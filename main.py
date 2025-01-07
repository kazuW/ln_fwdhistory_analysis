from openai import OpenAI
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import json
import gradio as gr
import pandas as pd
import sys
from datetime import datetime, timezone, timedelta

load_dotenv()

"""
class Channel(BaseModel):
    alias: str
    ave_fee: float
    sigma: float

class ChannelDetail(BaseModel):
    peer_alias: str
    input_peer: list[Channel]
    output_peer: list[Channel]

class ChannelList(BaseModel):
    channel_peer: list[ChannelDetail]
"""

class channel:
    def __init__(self, alias: str, ave_fee: float, ave_min: float, ave_max: float, amt_sat: float, count: int):
        self.alias = alias
        self.ave_fee = ave_fee
        self.ave_min = ave_min
        self.ave_max = ave_max
        self.amt_sat = amt_sat
        self.count = count

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)

    def to_dict(self):
        return self.__dict__

class channel_detail:
    def __init__(self, peer_alias: str, input_peer: list[channel], output_peer: list[channel], input_amt_sat: float, output_amt_sat: float, capacity: int, local_balance_ratio: float):
        self.peer_alias = peer_alias
        self.input_peer = input_peer
        self.output_peer = output_peer
        self.input_amt_sat = input_amt_sat
        self.output_amt_sat = output_amt_sat
        self.capacity = capacity
        self.local_balance_ratio = local_balance_ratio

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        return {
            "peer_alias": self.peer_alias,
            "input_peer": [peer.to_dict() for peer in self.input_peer],
            "output_peer": [peer.to_dict() for peer in self.output_peer],
            "input_amt_sat": self.input_amt_sat,
            "output_amt_sat": self.output_amt_sat,
            "capacity": self.capacity,
            "local_balance_ratio": self.local_balance_ratio
        }

class ChannelList:
    def __init__(self, channel_peer: list[channel_detail], analysis_result: str, start_time: str, end_time: str, channel_num: int):
        self.channel_peer = channel_peer
        self.start_time = start_time
        self.end_time = end_time
        self.analysis_result = analysis_result
        self.channel_num = channel_num

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        return {
            "channel_peer": [peer.to_dict() for peer in self.channel_peer],
            "analysis_result": self.analysis_result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "channel_num": self.channel_num
        }

# Unixタイムスタンプを日本時間に変換する関数
def convert_to_jst(unix_timestamp):
    # Unixタイムスタンプをdatetimeオブジェクトに変換
    dt_utc = datetime.fromtimestamp(int(unix_timestamp), tz=timezone.utc)
    # 日本時間（UTC+9）に変換
    dt_jst = dt_utc.astimezone(timezone(timedelta(hours=9)))
    return dt_jst.strftime('%Y-%m-%d %Y-%m-%d %H:%M:%S')

# fowarding履歴のJSONファイルを読み込む
def read_fwd_data(fwd_data_file):
    with open(fwd_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        fwd_data = data["forwarding_events"]
        #print("fowarding Data数 : ",len(fwd_data)

    # fowarding履歴のalias名がない場合、chan_id_in, chan_id_outに置き換える
    for i in range(0, len(fwd_data)):
        if "edge not found" in fwd_data[i]["peer_alias_in"]:
            fwd_data[i]["peer_alias_in"] = "n" + fwd_data[i]["chan_id_in"]
        if "edge not found" in fwd_data[i]["peer_alias_out"]:
            fwd_data[i]["peer_alias_out"] = "n" + fwd_data[i]["chan_id_out"]
            #print(fwd_data[i]["peer_alias_in"])
    
    return fwd_data

# 接続してるchanel状態のJSONファイルを読み込む
def read_channel_data(chan_data_file):
    with open(chan_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        channel_data = data["channels"]
        #print("channel Data数 : ",len(channel_data)
   
    return channel_data

# データを再構築する
def reStructure_data(fwd_data):
    data = []
    for i in range(0, len(fwd_data)):
        data.append(fwd_data[i]["peer_alias_in"])
        data.append(fwd_data[i]["peer_alias_out"])

    data = list(set(data))
    re_data = ChannelList(channel_peer=[], analysis_result="", start_time="", end_time="", channel_num=0)
    re_data.start_time = convert_to_jst(fwd_data[0]["timestamp"])
    re_data.end_time = convert_to_jst(fwd_data[-1]["timestamp"])

    for i in range(0, len(data)):
        re_data.channel_peer.append(channel_detail(data[i], [], [], 0, 0, 0, 0.0))
        for j in range(0, len(fwd_data)):
            if fwd_data[j]["peer_alias_in"] == data[i] and fwd_data[j]["peer_alias_out"] not in [peer.alias for peer in re_data.channel_peer[i].output_peer]:
                re_data.channel_peer[i].output_peer.append(channel(fwd_data[j]["peer_alias_out"], 0, float('inf'), float('-inf'), 0, 0))
            if fwd_data[j]["peer_alias_out"] == data[i] and fwd_data[j]["peer_alias_in"] not in [peer.alias for peer in re_data.channel_peer[i].input_peer]:
                re_data.channel_peer[i].input_peer.append(channel(fwd_data[j]["peer_alias_in"], 0, float('inf'), float('-inf'), 0, 0))
 
    for i in range(0, len(re_data.channel_peer)):
        for j in range(0, len(fwd_data)):
            if fwd_data[j]["peer_alias_in"] == re_data.channel_peer[i].peer_alias:
                for k in range(0, len(re_data.channel_peer[i].output_peer)):
                    if fwd_data[j]["peer_alias_out"] == re_data.channel_peer[i].output_peer[k].alias:
                        count = re_data.channel_peer[i].output_peer[k].count + 1
                        before_fee = re_data.channel_peer[i].output_peer[k].ave_fee
                        ave_fee = float(fwd_data[j]["fee"])
                        re_data.channel_peer[i].output_peer[k].ave_fee = (before_fee * (count - 1) + ave_fee) / count
                        re_data.channel_peer[i].output_peer[k].count = count
                        re_data.channel_peer[i].output_peer[k].amt_sat += float(fwd_data[j]["amt_in"])
                        if count == 1:
                            re_data.channel_peer[i].output_peer[k].ave_min = ave_fee
                        else:
                            re_data.channel_peer[i].output_peer[k].ave_min = min(re_data.channel_peer[i].output_peer[k].ave_min, float(fwd_data[j]["fee"]))
                        re_data.channel_peer[i].output_peer[k].ave_max = max(re_data.channel_peer[i].output_peer[k].ave_max, float(fwd_data[j]["fee"]))      
            if fwd_data[j]["peer_alias_out"] == re_data.channel_peer[i].peer_alias:
                for k in range(0, len(re_data.channel_peer[i].input_peer)):
                    if fwd_data[j]["peer_alias_in"] == re_data.channel_peer[i].input_peer[k].alias:
                        re_data.channel_peer[i].input_peer[k].count += 1
                        re_data.channel_peer[i].input_peer[k].amt_sat += float(fwd_data[j]["amt_out"])

    for i in range(0, len(re_data.channel_peer)):
        re_data.channel_peer[i].input_amt_sat = sum([peer.amt_sat for peer in re_data.channel_peer[i].input_peer])
        re_data.channel_peer[i].output_amt_sat = sum([peer.amt_sat for peer in re_data.channel_peer[i].output_peer])

    re_data.channel_num = len(re_data.channel_peer)

    return re_data

def print_data(re_data):

    print("##################################################")
    for i in range(0, len(re_data.channel_peer)):
        print(re_data.channel_peer[i].peer_alias, re_data.channel_peer[i].capacity, re_data.channel_peer[i].local_balance_ratio, re_data.channel_peer[i].input_amt_sat, re_data.channel_peer[i].output_amt_sat)
        for j in range(0, len(re_data.channel_peer[i].input_peer)):
            print("  input_peer: ", re_data.channel_peer[i].input_peer[j].alias, re_data.channel_peer[i].input_peer[j].ave_fee, re_data.channel_peer[i].input_peer[j].amt_sat)
        for j in range(0, len(re_data.channel_peer[i].output_peer)):
            print("  output_peer: ", re_data.channel_peer[i].output_peer[j].alias, re_data.channel_peer[i].output_peer[j].ave_fee, re_data.channel_peer[i].output_peer[j].amt_sat)

    print("##################################################")
    print("start_time: ", re_data.start_time)
    print("end_time: ", re_data.end_time)
    print("channel number: ", re_data.channel_num)

# データにチャンネル情報を追加する
def add_other_data(re_data, channel_data):
    for i in range(0, len(re_data.channel_peer)):
        for j in range(0, len(channel_data)):
            if channel_data[j]["peer_alias"][:10] == channel_data[j]["remote_pubkey"][:10]:
                channel_data[j]["peer_alias"] = "n" + channel_data[j]["chan_id"]
            if re_data.channel_peer[i].peer_alias == channel_data[j]["peer_alias"]:
                re_data.channel_peer[i].capacity += int(channel_data[j]["capacity"])
                re_data.channel_peer[i].local_balance_ratio = float(channel_data[j]["local_balance"]) / float(channel_data[j]["capacity"]) * 100
                re_data.channel_peer[i].local_balance_ratio = round(re_data.channel_peer[i].local_balance_ratio, 2)
    return re_data

"""
# channel_reasoning をデータフレームに変換
def convert_to_dataframe(channel_reasoning):
    data = []
    for detail in channel_reasoning.channel_peer:
        for peer in detail.input_peer:
            data.append([detail.peer_alias, "input", peer.alias, peer.ave_fee, peer.sigma])
        for peer in detail.output_peer:
            data.append([detail.peer_alias, "output", peer.alias, peer.ave_fee, peer.sigma])
    df = pd.DataFrame(data, columns=["Peer Alias", "Type", "Channel Alias", "Average Fee", "Sigma"])
    return df
"""
"""
# channel_reasoning をデータフレームに変換
def convert_to_dataframe(channel_reasoning):
    data = []
    for detail in channel_reasoning.channel_peer:
        input_details = "\n".join([f"{peer.alias} (Fee: {peer.ave_fee}, Sigma: {peer.sigma})" for peer in detail.input_peer])
        output_details = "\n".join([f"{peer.alias} (Fee: {peer.ave_fee}, Sigma: {peer.sigma})" for peer in detail.output_peer])
        data.append([detail.peer_alias, input_details, output_details])
    df = pd.DataFrame(data, columns=["Channel Peer", "Input Channels", "Output Channels"])
    return df

# Gradioインターフェースを作成
def analyze_forwarding_history():
    global channel_reasoning
    df = convert_to_dataframe(channel_reasoning)
    return df

iface = gr.Interface(
    fn=analyze_forwarding_history,
    inputs=[],
    outputs=gr.DataFrame(),
    title="Forwarding History Analysis",
    description="This tool analyzes the forwarding history and provides insights."
)
"""

def main():
    fwd_data = read_fwd_data("./history20250107-1w.json")
    re_data = reStructure_data(fwd_data)
    channel_data = read_channel_data("./listchannels.json")
    re_data = add_other_data(re_data, channel_data)

    print("データ期間: ", re_data.start_time, " ~ ", re_data.end_time)

    # re_dataを文字列に変換
    re_data_json_p = json.dumps(re_data.to_dict(), ensure_ascii=False, indent=2)
    re_data_json = json.dumps(re_data.to_dict(), ensure_ascii=False, separators=(',', ':'))
    #re_data_str = str(re_data)

    #print("---------------------------------")
    #with open('re_data_str.txt', 'w', encoding='utf-8') as file:
    #    file.write(re_data_str)
    print("-------------json data file write!! --------------------")
    with open('re_data_json.txt', 'w', encoding='utf-8') as file:
        file.write(re_data_json_p)
    #print("---------------------------------")

    sys.exit()
    # OpenAI APIを利用して、fowarding履歴の解析を行う
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": f""" \
                "channel_peer"は"channel_detail"のListです。\
                "channel_detail"にはnode名"peer_alias"とnodeに対する入力node名一覧"input_peer"出力node名一覧"output_peer"があります。 \
                入力node一覧にはそれぞれのnodeの総流入量"amt_sat"が示してあります。 \
                出力node一覧にはそれぞれのnodeの総流出量"amt_sat"が示してあります。\
                入力node一覧にはそれぞれのnodeの総流入量時の平均手数料"fee_ave"が示してあります。\
                出力node一覧にはそれぞれのnodeの総流出量時の平均手数料"fee_ave"が示してあります。 \
                "channel_detail"にはすべての入力nodeからの総流入量"input_amt_sat"が示してあります。\
                "channel_detail"にはすべての出力nodeからの総流出量"output_amt_sat"が示してあります。\
                "channel_detail"にはnode名"peer_alias"の現在の総容量に対するLocal balanceの割合が"local_balance_ratio"として示してあります。\

                以上を前提として以下のjson形式の入力をもとに次の考察をしてください。\n\n{re_data_json} \
                日本語に翻訳してください。

                1.各チャンネルの出力nodeの平均手数料が高く総流出量が多いチャンネルはありますか?
                    （このようなチャンネル群はより多くの手数料を稼ぎます。）
                2.各チャンネルの総流出量や総流入量が偏って多くLocal balanceの割合が偏ってしまうチャンネルはありますか?
                    (このようなチャンネルは片方に資金が貯まりやすく双方向に動きにくいnodeです。)
                3.各チャンネルのLocal balanceの割合が30%-70%に近く総流出量と総流入量が多いnodeはありますか?
                    （このようなチャンネルは双方向に流れることにより多くの資金を動かしており、より多くの手数料を稼げるため良いnodeといえます。）

                最後に各チャンネルの特性に基づいて、手数料の稼ぎやすさや資金の流れの偏りを分析してください。
                """,
            }
        ],
        temperature=0, 
        #response_format=ChannelList,
    )

    #print("##################################################")
    channel_reasoning = response.choices[0].message.content
    #channel_reasoning = response.choices[0].message.parsed
    print(channel_reasoning)  # デバッグ: channel_reasoning を表示
    #print("##################################################")

    with open('response.txt', 'w', encoding='utf-8') as file:
        file.write(channel_reasoning)


if __name__ == '__main__':
    main()
    #iface.launch()
