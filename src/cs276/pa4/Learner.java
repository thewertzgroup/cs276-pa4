package cs276.pa4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.pa4.Learner.Features;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

public abstract class Learner {
	
	private static Map<String, Map<Query,List<Document>>> queryMaps;
	static {
		queryMaps = new HashMap<>();
	}
	private static Map<String, Map<String, Map<String, Double>>> relMaps;
	static {
		relMaps = new HashMap<>();
	}
	
	public static Map<Query,List<Document>> getQueryMap(String data_file)
	{
		Map<Query,List<Document>> queryMap;
		
		queryMap = queryMaps.get(data_file);
		if (queryMap == null)
		{
			try {
				queryMap = Util.loadTrainData(data_file);
			} catch (Exception e) {
				throw new RuntimeException("Unable to load data file: " + data_file, e);
			}
			queryMaps.put(data_file, queryMap);
		}
		return queryMap;
	}
	
	public static Map<String, Map<String, Double>> getRelMap(String rel_file)
	{
		Map<String, Map<String, Double>> relMap;
		
		relMap = relMaps.get(rel_file);
		if (relMap == null)
		{
			try {
				relMap = Util.loadRelData(rel_file);
			} catch (IOException e) {
				throw new RuntimeException("Unable to load relevance file: " + rel_file, e);
			}
			relMaps.put(rel_file,  relMap);
		}
		return relMap;
	}
	
	public /*static*/ enum Features
	{
		BM25,			// BM25 Ranking
		SmallWindow,	// Smallest Window
		PageRank		// PageRank
	}
	
	private static Map<Map<Query,List<Document>>, BM25Scorer> bm25Scorers;
	static {
		bm25Scorers = new HashMap<>();
	}
	private static Map<Map<Query,List<Document>>, SmallestWindowScorer> smallestWindowScorers;
	static {
		smallestWindowScorers = new HashMap<>();
	}
	
	public static BM25Scorer getBM25Scorer(Map<Query,List<Document>> queryMap)
	{
		BM25Scorer scorer;
		
		scorer = bm25Scorers.get(queryMap);
		if (scorer == null)
		{
			scorer = new BM25Scorer(idfs, queryMap);
			bm25Scorers.put(queryMap,  scorer);
		}
		return scorer;
	}
	
	public static SmallestWindowScorer getSmallestWindowScorer(Map<Query,List<Document>> queryMap)
	{
		SmallestWindowScorer scorer;
		
		scorer = smallestWindowScorers.get(queryMap);
		if (scorer == null)
		{
			scorer = new SmallestWindowScorer(idfs, queryMap);
			smallestWindowScorers.put(queryMap,  scorer);
		}
		return scorer;
	}
	
    protected /*static*/ Map<Query,List<Document>> queryMap = null;
    protected /*static*/ Map<String, Map<String, Double>> relMap = null;
	
	protected /*static*/ AScorer bm25Scorer = null;
	
	protected /*static*/ AScorer smallestWindowScorer = null;
	
	protected /*static*/ List<Features> features = new ArrayList<>();
/*	
	public static void reset()
	{
		System.err.println("\n##### Learner RESET #####\n\n");
		queryMap = null;
		relMap = null;
		bm25Scorer = null;
		smallestWindowScorer = null;
	}
*/	
	
	protected static Map<String, Double> idfs = null;

	public static void setIDFs(Map<String, Double> idfs) 
	{
		Learner.idfs = idfs;
	}
	public static Map<String, Double> getIDFs()
	{
		return Learner.idfs;
	}

	public void setFeatures(List<Features> features) 
	{
		this.features = features;
	}

	public /*static*/ double[] getTFIDFVector(Document doc, Query query)
	{
		Map<String, Map<String, Double>> tfVectors = bm25Scorer.getDocTermFreqs(doc, query);
		Map<String, Double> idfVector = Util.getIDFVector(query, idfs);
		
		double[] instance = new double[6 + features.size()];
		for (int i=0; i<instance.length; i++) instance[i] = -1;

		// Iterate over tf vectors, and calculate field tf-idf
		for (String type : tfVectors.keySet())
		{
			Double score = bm25Scorer.dotVectors(tfVectors.get(type), idfVector);
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

		int index = 5;
		if (features.contains(Features.BM25))
		{
			instance[index++] = bm25Scorer.getSimScore(doc, query);
		}
		if (features.contains(Features.SmallWindow))
		{
			instance[index++] = smallestWindowScorer.getSimScore(doc, query);
		}
		if (features.contains(Features.PageRank))
		{
			instance[index++] = doc.page_rank;
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
