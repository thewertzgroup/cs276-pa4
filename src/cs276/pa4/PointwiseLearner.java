package cs276.pa4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner 
{

	@Override
	public Instances extract_train_features(String train_data_file, String train_rel_file, Map<String, Double> idfs) 
	{
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		/* Add data */
		Map<Query,List<Document>> queryMap = null;
		Map<String, Map<String, Double>> relMap = null;
		try 
		{
			queryMap = Util.loadTrainData(train_data_file);
			relMap 	 = Util.loadRelData(train_rel_file);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			throw new RuntimeException("Unable to load signal data, or relevance data.", e);
		}
		
		// Iterate over queries / documents and compute five-dimensional vector of tf-idf scores.	
		AScorer scorer = new BM25Scorer(idfs, queryMap);
		
		for (Query query : queryMap.keySet())
		{
			List<Document> docs = queryMap.get(query);
			for (Document doc : docs)
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
				
				// TODO: ADD RELEVANCE SCORE (TARGET VARIABLE) HERE.
				instance[5] = relMap.get(query.query).get(doc.url);
				
				Instance inst = new DenseInstance(1.0, instance); 
				dataset.add(inst);
			}
		}
		
		return dataset;
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
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) 
	{
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

	
	@Override
	public Map<String, List<String>> testing(TestFeatures tf, Classifier model) 
	{
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

}
