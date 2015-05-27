package cs276.pa4;

import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {
	
	protected static AScorer scorer = null;
	
	protected static Map<String, Double> idfs = null;
	
	public void setIDFs(Map<String, Double> idfs) 
	{
		this.idfs = idfs;
	}
	
	public Map<String, Double> getIDFs()
	{
		return this.idfs;
	}

	public static double[] getTFIDFVector(Document doc, Query query)
	{
		Map<String, Map<String, Double>> tfVectors = scorer.getDocTermFreqs(doc, query);
		Map<String, Double> idfVector = Util.getIDFVector(query, idfs);
		
		double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

		// Iterate over tf vectors, and calculate field tf-idf
		for (String type : tfVectors.keySet())
		{
			Double score = scorer.dotVectors(tfVectors.get(type), idfVector);
			
			// "url","title","body","header","anchor"
			if (type.equals("url"))
			{
				instance[0] = score;
			}
			else if (type.equals("title"))
			{
				instance[1] = score;
			}
			else if (type.equals("body"))
			{
				instance[2] = score;
			}
			else if (type.equals("header"))
			{
				instance[3] = score;
			}
			else if (type.equals("anchor"))
			{
				instance[4] = score;
			}
			else
			{
				throw new RuntimeException("Unsupported type in PointwiseLearner.");
			}
		}
		
		return instance;
	}

	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file);
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);

}
