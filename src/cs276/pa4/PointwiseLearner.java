package cs276.pa4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.pa4.AScorer;
import cs276.pa4.Pair;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception {
		
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
		
		// load the training and the relevance score data
		Map<Query,List<Document>> queryDict = Util.loadTrainData(train_data_file);
		Map<String, Map<String, Double>> relevanceDict = Util.loadRelData(train_rel_file); 
		// create a scorer to extract features
		AScorer extractor = new tf_idfExtractor(idfs);
		Map<String, Double> tf_idfs; 
		// go over every (query, document) instance in the training data, compute the features, and add it to the data set	
		for (Query query : queryDict.keySet()) {
			Map<String, Double> queryRelMap = relevanceDict.get(query.query);			 
			// Loop through the documents for query, getting scores
			for (Document doc : queryDict.get(query)) {
				// for this query document pair 
				// 1. get the tf-idf features for every one of the five fields 
				tf_idfs = extractor.getFeatures(doc, query);
				double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
				int fieldInd = 0; 
				for(String field: tf_idfs.keySet())
					instance[fieldInd++] = tf_idfs.get(field); 
				// 2. Get the relevance score				
				instance[5] = queryRelMap.get(doc.url); 
				// 3. add the data instance for this query document pair to the data set				
				Instance inst = new DenseInstance(1.0, instance); 
				dataset.add(inst);
			}
		}								
		
		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) throws Exception {
		/*
		 * @TODO: Your code here
		 */
		LinearRegression model = new LinearRegression();
		model.buildClassifier(dataset);
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) throws Exception {
		/*
		 * @TODO: Your code here
		 */
		
		TestFeatures myTestFeatures = new TestFeatures();
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		myTestFeatures.features = new Instances("test_dataset", attributes, 0);
		myTestFeatures.index_map = new HashMap<String, Map<String, Integer>>(); 
		/* Set last attribute as target */
	//	myTestFeatures.features.setClassIndex(myTestFeatures.features.numAttributes() - 1);
		
		// load the training and the relevance score data
		Map<Query,List<Document>> queryDict = Util.loadTrainData(test_data_file);
		// create a scorer to extract features
		AScorer extractor = new tf_idfExtractor(idfs);
		Map<String, Double> tf_idfs; 
		// go over every (query, document) instance in the training data, compute the features, and add it to the data set
		for (Query query : queryDict.keySet()) {
		//	System.out.println("query: " + query.query);
			Map<String, Integer> queryDocFeatureMap = new HashMap<String, Integer>(); 
			// Loop through the documents for query, getting scores
			for (Document doc : queryDict.get(query)) {
				// for this query document pair 
				// 1. get the tf-idf features for every one of the five fields 
				tf_idfs = extractor.getFeatures(doc, query);
				double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
				int fieldInd = 0; 
				for(String field: tf_idfs.keySet())
					instance[fieldInd++] = tf_idfs.get(field);
	
	
								
				// 2. add the data instance for this query document pair to the data set				
				Instance inst = new DenseInstance(1.0, instance); 
				
				
				myTestFeatures.features.add(inst);
		//		System.out.print((myTestFeatures.features.size()-1) + " "); 
				queryDocFeatureMap.put(doc.url, myTestFeatures.features.size()-1); 								
			}
		//	System.out.print("\n"); 
			myTestFeatures.index_map.put(query.query, queryDocFeatureMap); 
		}								
		
		return myTestFeatures;
				
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) throws Exception {
		/*
		 * @TODO: Your code here
		 */
		Map<String,List<String>> ranked_queries = new HashMap<String, List<String>>(); 		
		Instances test_dataset = tf.features; /* The dataset you built in Step 3 */
		double prediction;
		int queryDocInd = 0; 
		Map<String, Integer> queryDocFeatureMap; 
		for(String q: tf.index_map.keySet())
		{ 
		//	System.out.println("testing query: " + q); 
			List<Pair<String,Double>> urlAndScores = new ArrayList<Pair<String,Double>>();
			queryDocFeatureMap = tf.index_map.get(q);
			for(String url: queryDocFeatureMap.keySet())
			{ 
				queryDocInd = queryDocFeatureMap.get(url); 
		//		System.out.print(queryDocInd+" "); 
				prediction = model.classifyInstance(test_dataset.instance(queryDocInd));
				urlAndScores.add(new Pair<String, Double>(url, prediction)); 				
			} 
		//	System.out.print("\n"); 
			// Sort urls for query based on scores
			Collections.sort(urlAndScores, new Comparator<Pair<String,Double>>() {
			@Override
			public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
				/*
				 * @//TODO : Your code here
				 */
				if (o2.getSecond().equals(o1.getSecond()))
					return o1.getFirst().compareTo(o2.getFirst()); 
				return o2.getSecond().compareTo(o1.getSecond());  
				
			}	
			});
		
			List<String> rankedUrl = new ArrayList<String>(); 
			for(Pair<String, Double> urlScorePair: urlAndScores)
			{ 
				rankedUrl.add(urlScorePair.getFirst()); 
			}
		
			ranked_queries.put(q, rankedUrl); 
		} 
	/*	double weights[] = ((LinearRegression)model).coefficients(); 
		System.out.println("The model weights are" );
		for(int i = 0; i< weights.length; i++)
			System.out.print(weights[i] + " ");
		System.out.print("\n");*/
		return ranked_queries;
	}

}
