from datetime import datetime
import pandas as pd


def extract_event_features(tweet, events_df):
    event_match = 0
    # Look for matching dates
    for event in events_df.index:
        if events_df.iloc[event]['start_date'] <= tweet <= events_df.iloc[event]['end_date']:
            event_match = 1
            event_name = events_df.iloc[event]['event_name']
            event_type = events_df.iloc[event]['event_type']
            days_since_start = (tweet - events_df.iloc[event]['start_date']).total_seconds()/(60*60*24)
            return {'event_match':1, 'event_name':event_name, 'event_type':event_type, 'days_since_start': days_since_start}
    # for ended, if no match - return {}
    return {'event_match':0}

def get_event_features(df, config):
    # First - convert df.created_at to timestamp
    df['created_at'] = df['created_at'].apply(lambda x: datetime.fromtimestamp(x))

    events_df = pd.read_csv('../../data/classifiers/events.csv', header=0, encoding='utf-8')
    events_df['start_date'] = events_df['start_date'].apply(lambda x: datetime.strptime(str(x), '%d/%m/%Y'))
    events_df['end_date'] = events_df['end_date'].apply(lambda x: datetime.strptime(str(x), '%d/%m/%Y'))

    # Extract event features into a single field (using dict)
    df['event_data'] = df.created_at.apply(lambda x: extract_event_features(x, events_df))

    # Split the dict to separated features
    df['in_event'] = df.event_data.apply(lambda x: x.get('event_match', 0))
    # Get event type
    df['event_type'] = df.event_data.apply(lambda x: x.get('event_type', 'No type'))
    df['days_since_start'] = df.event_data.apply(lambda x: x.get('days_since_start', 0))

    # Create dummy features for event type
    event_type_df = pd.get_dummies(df['event_type'])
    # In case non of the Tweets belong to an event type - create dummy feature with values = 0
    event_types = set(events_df['event_type'].tolist())
    event_types.add('No type')

    for e_type in event_types:
        if e_type not in event_type_df.columns:
            event_type_df[e_type] = 0

    # Concat results and drop unnecessary features
    df = pd.concat([event_type_df, df], axis=1)
    df = df.drop(columns=['event_data','event_type'], axis=1)

    return df