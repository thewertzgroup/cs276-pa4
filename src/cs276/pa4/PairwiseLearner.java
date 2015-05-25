package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  private ArrayList<Attribute> attributes = new ArrayList<Attribute>();
  
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
     //0. Build the attributes
   	 /* Build attributes list */
  	
  	 attributes.add(new Attribute("url_w"));
  	 attributes.add(new Attribute("title_w"));
  	 attributes.add(new Attribute("body_w"));
  	 attributes.add(new Attribute("header_w"));
  	 attributes.add(new Attribute("anchor_w"));
  	 attributes.add(new Attribute("relevance_score"));

    } catch (Exception e){
      e.printStackTrace();
    }
    model.setCost(1.0E-21);
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
    
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
      //0. Build the attributes
      /* Build attributes list */
   	
   	  attributes.add(new Attribute("url_w"));
   	  attributes.add(new Attribute("title_w"));
   	  attributes.add(new Attribute("body_w"));
   	  attributes.add(new Attribute("header_w"));
   	  attributes.add(new Attribute("anchor_w"));
   	  attributes.add(new Attribute("relevance_score"));

    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public Instances standardizeFeatures(Instances features) throws Exception
  { 
	  Standardize filter = new Standardize();	  
	  filter.setInputFormat(features);
	  Instances standardizedFeatures = Filter.useFilter(features, filter);
	  
	  return standardizedFeatures;
  }
  
  public Pair<Instances, Map <String, List<Pair<String, Integer>>>>  extractPointWiseFeatures(String test_data_file,
			 Map<String, Double> idfs, ArrayList<Attribute> attributes) throws Exception{
	  
	  Pair<Instances, Map <String, List<Pair<String, Integer>>>> featuresANDindices = extractPointWiseFeatures(test_data_file,
				"", idfs, attributes); 	
	  return featuresANDindices; 
}
  public Pair<Instances, Map <String, List<Pair<String, Integer>>>>  extractPointWiseFeatures(String train_data_file,
			String train_rel_file, Map<String, Double> idfs, ArrayList<Attribute> attributes) throws Exception{
	  	  
	  	Pair<Instances, Map <String, List<Pair<String, Integer>>>> featuresANDindices = null; 
	  
	  	Instances pointWiseFeatures = null;
	  	Map <String, List<Pair<String, Integer>>> indexMap = new HashMap<String, List<Pair<String, Integer>>>(); 
		
		pointWiseFeatures = new Instances("pre_train_dataset", attributes, 0);
				
		// load the training and the relevance score data
		Map<Query,List<Document>> queryDict = Util.loadTrainData(train_data_file);
		Map<String, Map<String, Double>> relevanceDict = null ;
		if(!train_rel_file.equals(""))
			relevanceDict = Util.loadRelData(train_rel_file); 
		// create a scorer to extract features
		AScorer extractor = new tf_idfExtractor(idfs);
		Map<String, Double> tf_idfs;
		Map<String, Double> queryRelMap = null;
		// go over every (query, document) instance in the training data, compute the features, and add it to the data set	
		for (Query query : queryDict.keySet()) {
			if(!train_rel_file.equals(""))
				queryRelMap = relevanceDict.get(query.query);	
			List<Pair<String, Integer>> queryDocFeatureList = new ArrayList<Pair<String, Integer>>();  
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
				if(!train_rel_file.equals(""))
					instance[5] = queryRelMap.get(doc.url); 
				// 3. add the data instance for this query document pair to the data set				
				Instance inst = new DenseInstance(1.0, instance); 
				pointWiseFeatures.add(inst);
				queryDocFeatureList.add(new Pair<String, Integer>(doc.url, pointWiseFeatures.size()-1)); 	
			}
			indexMap.put(query.query, queryDocFeatureList); 
		}
		
		featuresANDindices = new Pair<Instances, Map <String, List<Pair<String, Integer>>>>(pointWiseFeatures, indexMap); 
	  return featuresANDindices; 
  }
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception {
		/*
		 * @TODO: Your code here
		 */				
		// 1. Get the pointwise features  
		Pair<Instances, Map <String, List<Pair<String, Integer>>>>  featuresANDindices = 
				 extractPointWiseFeatures(train_data_file, train_rel_file, idfs, attributes);
		Instances pointWiseFeatures 					   = featuresANDindices.getFirst(); 
		Map <String, List<Pair<String, Integer>>> indexMap = featuresANDindices.getSecond();  
		
		// 2. Standardize the pointwise features
		Instances standardizedFeatures = standardizeFeatures(pointWiseFeatures); 
		
		// 3. Get the pairwise features 
		Instances pairWiseDataset = null;
		pairWiseDataset = new Instances("train_dataset", attributes, 0);
		
	//	System.out.println("number of attributes is " + pairWiseDataset.numAttributes()); 
		List<Pair<String, Integer>> queryDocFeatureList;
		int queryDocInd1 = 0, queryDocInd2 = 0; 
		double relevance1 = 0.0, relevance2 = 0.0;  
		int numPositive = 0, numNegative = 0;
		double label = 0;
		double flip  = 1; 
		for(String q: indexMap.keySet())
		{ 			
			queryDocFeatureList = indexMap.get(q);
			for(int ind1 = 0; ind1 < queryDocFeatureList.size(); ind1++)
				for( int ind2 = ind1+1; ind2 < queryDocFeatureList.size(); ind2++)
				{ 
					queryDocInd1 = queryDocFeatureList.get(ind1).getSecond();
					queryDocInd2 = queryDocFeatureList.get(ind2).getSecond(); 
					relevance1 = standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute("relevance_score"));
					relevance2 = standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute("relevance_score"));
					
					if(relevance1!=relevance2)
					{ 						
						double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
						label = Math.signum(relevance1 - relevance2);
						
						//balance the features: negative and positive instances must be balanced
						if(label*(numPositive - numNegative) >0) // label > 0 and numPositive > numNegative OR label < 0 and numPositive < numNegative 
							flip = -1; 
						else 
							flip = 1; 
						
						instance[0] = flip * ( standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute("url_w"))
								             - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute("url_w")));
						instance[1] = flip * ( standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute("title_w"))
							                 - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute("title_w")));
						instance[2] = flip * ( standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute("body_w"))
							                 - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute("body_w")));
						instance[3] = flip * ( standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute("header_w"))
						                     - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute("header_w")));					
						instance[4] = flip * ( standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute("anchor_w"))
						                     - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute("anchor_w")));
						instance[5] = flip*label; 
						
					//	System.out.print(instance[5] + " "); 
						if(instance[5]>0)
							numPositive++; 
						else 
							numNegative++;
						
						Instance inst = new DenseInstance(1.0, instance);
					
						pairWiseDataset.add(inst);
					}
				} 
		} 
		
		// check if we balanced the features: negative and positive instances must be balanced
		/*
		System.out.println("numPositive = " + numPositive);
		System.out.println("numNegative = " + numNegative); 
		*/
		//4. convert last attribute, the target/label, to categorical as we have classification not regression 
		NumericToNominal convert= new NumericToNominal();
	    String[] options= new String[2];
	    options[0]="-R";
	    options[1]="6";  //range of variables to make nominal: just the last one
        convert.setOptions(options);
	    convert.setInputFormat(pairWiseDataset);

	    Instances finalPairWiseDataset=Filter.useFilter(pairWiseDataset, convert);
	    finalPairWiseDataset.setClassIndex(pairWiseDataset.numAttributes() - 1);
	   /* System.out.println(" category " + finalPairWiseDataset.get(0).stringValue(finalPairWiseDataset.attribute("relevance_score"))); 
	    System.out.println(" category " + finalPairWiseDataset.get(1).stringValue(finalPairWiseDataset.attribute("relevance_score")));*/
		return finalPairWiseDataset;
		
	}

	@Override
	public Classifier training(Instances dataset) throws Exception {
		/*
		 * @TODO: Your code here
		 */		
		model.buildClassifier(dataset);
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) throws Exception {
		/*
		 * @TODO: Your code here
		 */
		// 1. Get the pointwise features  
		Pair<Instances, Map <String, List<Pair<String, Integer>>>>  featuresANDindices = 
				 extractPointWiseFeatures(test_data_file, idfs, attributes);
		Instances pointWiseFeatures 					   = featuresANDindices.getFirst(); 
		Map <String, List<Pair<String, Integer>>> indexMap = featuresANDindices.getSecond();  
		
		// 2. Standardize the pointwise features
		Instances standardizedFeatures = standardizeFeatures(pointWiseFeatures); 
		
		// 3. put the result in TestFeatures object
		TestFeatures myTestFeatures = new TestFeatures(); 
		myTestFeatures.features = standardizedFeatures; 
		myTestFeatures.index_map = new HashMap<String, Map<String, Integer>>(); 
		
		for (String q: indexMap.keySet())
		{ 
			Map<String, Integer> queryDocFeatureMap = new HashMap<String, Integer>(); 
			for(int ind = 0; ind < indexMap.get(q).size(); ind++ )
				queryDocFeatureMap.put(indexMap.get(q).get(ind).getFirst(), indexMap.get(q).get(ind).getSecond()); 
			myTestFeatures.index_map.put(q, queryDocFeatureMap); 
		}
		
		return myTestFeatures;
	}
	
	public double getScore(Instance inst, double weights[])
	{ 
		double score = 0.0;
		
	//	System.out.println("The model weights are" );
		int weightInd = 0; 
		for(weightInd = 0; weightInd< weights.length-1; weightInd++)
		{ 
		//	System.out.print(weights[i] + " ");
			score += weights[weightInd] * inst.value(weightInd); 
		}
		score += weights[weightInd];  
		
		return score; 
	}
	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) throws Exception {
		/*
		 * @TODO: Your code here
		 */
		
		System.out.print("\n");
		System.out.println( "C = "+ ((LibSVM)model).getCost()); 
		double weights[] = ((LibSVM)model).coefficients(); 
		Map<String,List<String>> ranked_queries = new HashMap<String, List<String>>(); 		
		Instances test_dataset = tf.features; /* The dataset you built in Step 3 */
		test_dataset.setClassIndex(test_dataset.numAttributes()-1);
		double score;
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
			//	score = model.classifyInstance(test_dataset.instance(queryDocInd));
				score = getScore(test_dataset.instance(queryDocInd), weights); 
			//	System.out.println("prediction " +  model.classifyInstance(test_dataset.instance(queryDocInd))); 
				urlAndScores.add(new Pair<String, Double>(url, score)); 				
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
		
		return ranked_queries;
	}

}
