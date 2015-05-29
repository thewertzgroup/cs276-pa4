package cs276.pa4;

import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {
	
	boolean withBM25, withSmallestWindow, withPageRank, withUrlLen, withTitleLen, withBodyLen, withHeaderLen, withAnchorLen, withUrlPDF; 	
	public Learner(boolean withBM25, boolean withSmallestWindow, boolean withPageRank, boolean withUrlLen, boolean withTitleLen, boolean withBodyLen, boolean withHeaderLen, boolean withAnchorLen, boolean withUrlPDF)
	{ 
		this.withBM25 = withBM25;
		this.withSmallestWindow = withSmallestWindow; 
		this.withPageRank = withPageRank;
		
		this.withUrlLen = withUrlLen; 		
		this.withTitleLen = withTitleLen; 
		this.withBodyLen = withBodyLen; 
		this.withHeaderLen = withHeaderLen; 
		this.withAnchorLen = withAnchorLen;
		
		this.withUrlPDF = withUrlPDF; 
	}
	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file, Map<String,Double> idfs) throws Exception;

	/* Train the model */
	public abstract Classifier training (Instances dataset) throws Exception;
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file, Map<String,Double> idfs) throws Exception;
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model) throws Exception;
}
