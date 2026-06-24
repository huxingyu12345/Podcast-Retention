import pandas as pd
import os

DATA_DIR = r'/mnt/workspace/xiaoyuzhou/dataset_algo2025'

def analyze_cold_start():
    print("Loading datasets...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    print("\n--- Cold Start Analysis ---")
    
    # 1. User Cold Start
    train_uids = set(train['uid'].unique())
    test_uids = set(test['uid'].unique())
    
    new_users = test_uids - train_uids
    num_new_users = len(new_users)
    total_test_users = len(test_uids)
    
    test_rows_with_new_users = test[test['uid'].isin(new_users)].shape[0]
    total_test_rows = test.shape[0]

    print(f"Unique Users in Train: {len(train_uids)}")
    print(f"Unique Users in Test: {total_test_users}")
    print(f"New Users in Test (Cold Start): {num_new_users} ({num_new_users/total_test_users*100:.2f}% of unique test users)")
    print(f"Test Rows with New Users: {test_rows_with_new_users} ({test_rows_with_new_users/total_test_rows*100:.2f}% of total test rows)")

    # 2. Episode Cold Start
    train_episodes = set(train['episode_id'].unique())
    test_episodes = set(test['episode_id'].unique())
    
    new_episodes = test_episodes - train_episodes
    num_new_episodes = len(new_episodes)
    total_test_episodes = len(test_episodes)
    
    test_rows_with_new_episodes = test[test['episode_id'].isin(new_episodes)].shape[0]

    print(f"\nUnique Episodes in Train: {len(train_episodes)}")
    print(f"Unique Episodes in Test: {total_test_episodes}")
    print(f"New Episodes in Test (Cold Start): {num_new_episodes} ({num_new_episodes/total_test_episodes*100:.2f}% of unique test episodes)")
    print(f"Test Rows with New Episodes: {test_rows_with_new_episodes} ({test_rows_with_new_episodes/total_test_rows*100:.2f}% of total test rows)")

    # 3. Double Cold Start (Both User and Episode are new)
    test_double_cold = test[test['uid'].isin(new_users) & test['episode_id'].isin(new_episodes)].shape[0]
    print(f"\nTest Rows with Double Cold Start (New User & New Episode): {test_double_cold} ({test_double_cold/total_test_rows*100:.2f}% of total test rows)")

    print("\n--- Summary ---")
    if test_rows_with_new_users > 0 or test_rows_with_new_episodes > 0:
        print("Result: Cold start issue EXISTS in the test set.")
        print("Recommendation: Use user/episode features (age, sex, category, title embeddings) to generalize for unseen IDs.")
    else:
        print("Result: No cold start issue detected. ID-based features should be very strong.")

if __name__ == '__main__':
    analyze_cold_start()
