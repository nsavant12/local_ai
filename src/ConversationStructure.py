from datetime import datetime, timedelta
import pandas as pd

def build_reply_examples(
    mssgs: list[dict],
    self_sender: str = "Me",
    max_context_turns: int = 6,
) -> list[dict]:
    if max_context_turns <=0:
        raise ValueError("max_context_turns needs to be positive")

    #Consecutive mssgs from same sender are treated as 1 mssg (combined)
    turns = []
    for mssg in mssgs:
        sender = mssg["sender"]
        txt = mssg["message"]

        if not txt.strip():
            continue
        if turns and turns[-1]["sender"] == sender:
            turns[-1]["text"] += "\n" + txt
        else:
            turns.append({"sender": sender, "text": txt})

    #Each turn is a target w/ preceding context
    ex = []

    for idx, turn in enumerate(turns):
        if turn["sender"] != self_sender:
            continue
        if idx == 0:
            continue
    
        start_context = max(0, idx - max_context_turns)
        context = [prev_turn.copy() for prev_turn in turns[start_context : idx]]
        ex.append({"context": context, "reply": turn["text"]})

    return ex

def split_sessions(
    mssgs: list[dict],
    max_time: timedelta = timedelta(hours=1)
) -> list[list[dict]]:

    if max_time <= timedelta(0):
        raise ValueError("max_time <= 0. need > 0")
    
    sessions = []
    curr_session = []
    prev_time = None

    for mssg in mssgs:
        timestamp = mssg["timestamp"]

        if pd.isna(timestamp):
            raise ValueError("Cant split sessions w/o timestamps")
        
        if not isinstance(timestamp,datetime):
            raise ValueError("Timestamps must alr be parsed datetime vals")
        
        if prev_time is not None:
            gap = timestamp - prev_time
            if gap < timedelta(0):
                raise ValueError("Mssgs must be in chronological order")
            
            if gap > max_time:
                sessions.append(curr_session)
                curr_session = []
        curr_session.append(mssg.copy())
        prev_time = timestamp

    if curr_session:
        sessions.append(curr_session)
        
    return sessions
