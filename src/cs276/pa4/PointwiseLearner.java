package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner 
{
	private TestFeatures extract_dataset(String data_file)
	{
		return extract_dataset(data_file, null);
	}
	
	
	public /*static*/ TestFeatures extract_dataset(String data_file, String relevance_file)
	{
		TestFeatures testFeatures = new TestFeatures();
		testFeatures.index_map = new HashMap<>();
System.err.println("Pointwise Learner features: " + features);		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		if (features.contains(Features.BM25))
		{
			attributes.add(new Attribute("bm25"));

		}
		if (features.contains(Features.SmallWindow))
		{
			attributes.add(new Attribute("smallest_window"));

		}
		if (features.contains(Features.PageRank))
		{
			attributes.add(new Attribute("page_rank"));

		}
		attributes.add(new Attribute("relevance_score"));
		
		testFeatures.features = new Instances(relevance_file != null ? "train_dataset" : "test_dataset", attributes, 0);
		
		if (relevance_file != null)
		{
			/* Set last attribute as target */
			testFeatures.features.setClassIndex(testFeatures.features.numAttributes() - 1);

		}

		/* Add data */
		queryMap = Learner.getQueryMap(data_file);
		relMap 	 = relevance_file != null ? Learner.getRelMap(relevance_file) : null;

		// Set scorers for base class.
		bm25Scorer = Learner.getBM25Scorer(queryMap);
		smallestWindowScorer = Learner.getSmallestWindowScorer(queryMap);
		
		for (Query query : queryMap.keySet())
		{
			// Maps the URL to the row index of the test matrix to retrieve for prediction.
			Map<String, Integer> urlIndexMap = new HashMap<>(); 

			List<Document> docs = queryMap.get(query);
			for (Document doc : docs)
			{
				double[] instance = getTFIDFVector(doc, query);
				
				if (relMap != null)
				{
					// ADD RELEVANCE SCORE (TARGET VARIABLE) HERE.
					instance[testFeatures.features.numAttributes() - 1] = relMap.get(query.query).get(doc.url);
				}
				
				Instance inst = new DenseInstance(1.0, instance); 
				testFeatures.features.add(inst);
				
				urlIndexMap.put(doc.url, testFeatures.features.size()-1);
			}
			
			testFeatures.index_map.put(query.query,  urlIndexMap);
		}

		return testFeatures;
	}
	
	@Override
	public Instances extract_train_features(String train_data_file, String train_rel_file) 
	{
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
		return extract_dataset(train_data_file, train_rel_file).features;
	}

	
	@Override
	public Classifier training(Instances dataset) 
	{
		/*
		 * @TODO: Your code here
		 */
		LinearRegression classifier = new LinearRegression();
		
		try 
		{
			classifier.buildClassifier(dataset);
		} catch (Exception e) 
		{
			e.printStackTrace();
			throw new RuntimeException("Exception in PointwiseLearner on training.", e);
		}

		return classifier;
	}

	
	@Override
	public TestFeatures extract_test_features(String test_data_file) 
	{
		/*
		 * @TODO: Your code here
		 */
	
		return extract_dataset(test_data_file);
	}

	
	@Override
	public Map<String, List<String>> testing(TestFeatures tf, Classifier model) 
	{
		/*
		 * @TODO: Your code here
		 */
		Map<String, List<String>> rankedQueries = new HashMap<String, List<String>>();
		
		for (String query : tf.index_map.keySet())
		{
			
			Map<String, Integer> urlIndexMap = tf.index_map.get(query);
			
			Map<Double, String> results = new TreeMap<>(Collections.reverseOrder());
			
			for (String url : urlIndexMap.keySet())
			{
				Integer index = urlIndexMap.get(url);
				double result;
				try 
				{
					result = model.classifyInstance(tf.features.get(index));
				} 
				catch (Exception e) 
				{
					e.printStackTrace();
					throw new RuntimeException("Exception in classifyInstance.", e);
				}
				results.put(result, url);
			}
			
			rankedQueries.put(query, new ArrayList<String>(results.values()));
		}

		return rankedQueries;
	}

}
