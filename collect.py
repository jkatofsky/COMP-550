import praw
import math
import re
import pandas as pd
import csv
import argparse
from datetime import datetime
import logging

from secrets import ID, SECRET

reddit = praw.Reddit(
    client_id=ID,
    client_secret=SECRET,
    user_agent="COMP-550",
)

# Uncomment if you want to see an ungodly amount of HTTP request logging
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# for logger_name in ("praw", "prawcore"):
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.DEBUG)
#     logger.addHandler(handler)

def clean_comment(comment):
    comment = comment.replace('\n', ' ')
    return re.sub(r'http\S+', '', comment)


def get_comments(subreddit, num_posts, post_sort, time_filter=None,
                    max_comments_per_post=None, save_to_filename=None, with_user_flair_only=False):

    if not time_filter:
        time_filter = 'all'
    if not max_comments_per_post:
        max_comments_per_post = math.inf

    if post_sort == 'new':
        posts = reddit.subreddit(subreddit).new(limit=num_posts)
    elif post_sort == 'hot':
        posts = reddit.subreddit(subreddit).hot(limit=num_posts)
    elif post_sort == 'top':
        posts = reddit.subreddit(subreddit).top(limit=num_posts, time_filter=time_filter)
    
    comments = []

    for num, post in enumerate(posts, start=1):

        print(f'Collecting comments for post {num}...')
        post.comment_sort = 'top'
        # this was waaaay too slow to be useful
        # post.comments.replace_more(limit=max_comments_per_post)
        collected = 0
        comment_queue = post.comments[:]
        while comment_queue and collected < max_comments_per_post:
            comment = comment_queue.pop(0)

            if hasattr(comment, 'body') and comment.body \
                and (not with_user_flair_only or comment.author_flair_text):
                comments.append((clean_comment(comment.body),
                                comment.author_flair_text,
                                comment.author.name,
                                datetime.utcfromtimestamp(int(comment.created_utc)).strftime("%m/%d/%Y, %H:%M:%S"),
                                comment.link_id,
                                comment.id))
                collected += 1
            if hasattr(comment, 'replies'):
                comment_queue.extend(comment.replies)
        print(f'{len(comments)} comments collected.')

    df = pd.DataFrame(comments, columns=['comment', 'flair', 'username', 'time', 'post_id', 'comment_id'])
    if save_to_filename:
        df.to_csv(save_to_filename, index=False, quoting=csv.QUOTE_ALL)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--num_posts', type=int, required=True)
    parser.add_argument('--post_sort', type=str, required=True)
    parser.add_argument('--output_filename', type=str, required=True)
    parser.add_argument('--time_filter', type=str, default=None)
    parser.add_argument('--max_comments_per_post', type=int, default=None)
    parser.add_argument('--with_user_flair_only', type=bool, default=False)
    args = parser.parse_args()

    get_comments(subreddit=args.subreddit,
                 num_posts=args.num_posts, 
                 post_sort=args.post_sort,
                 time_filter=args.time_filter,
                 max_comments_per_post=args.max_comments_per_post,
                 save_to_filename=args.output_filename,
                 with_user_flair_only=args.with_user_flair_only)