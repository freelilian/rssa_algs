# rssa_algs
Environment requirements were exported in lenskit11.yml.

APIs of the RSSA recommendation lists was implemented in RSSA_recommendations.py:
list1 - get_RSSA_topN();
list2 - get_RSSA_hate_items();
list3 - get_RSSA_hip_items();
list4 - get_RSSA_no_clue_items();
list5 - get_RSSA_controversial_items();
	
Each method take a user ID as input, return movie IDs in an numpy.ndarray.
For testing purpose, available user IDs are: Bart, Daricia, Sushmita, Shahan, Aru, Mitali, Yash .
		
For testing data files, please download it from google drive separately.