package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
  final String [] classLabels = {"-1", "1"}; 
  public PairwiseLearner(boolean isLinearKernel){
		super(false, false, false, false, false, false, false, false, false);
	    try{
	      model = new LibSVM();
	     //0. Build the attributes
	   	 /* Build attributes list */
	  	
	  	 attributes.add(new Attribute("url_w"));
	  	 attributes.add(new Attribute("title_w"));
	  	 attributes.add(new Attribute("body_w"));
	  	 attributes.add(new Attribute("header_w"));
	  	 attributes.add(new Attribute("anchor_w"));

	  	 
	  	

	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    
	  //  double C = 1.0; 
	  //  model.setCost(C);
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	      
	      
	    }
	    else 
	    { 
	    	model.setCost(2.0);
		    model.setGamma(0.5); // only matter for RBF kernel 
	    }
	    
	  }
  public PairwiseLearner(boolean isLinearKernel, boolean withBM25, boolean withSmallestWindow, boolean withPageRank, boolean withUrlLen, boolean withTitleLen, boolean withBodyLen, boolean withHeaderLen, boolean withAnchorLen, boolean withUrlPDF){
	super(withBM25, withSmallestWindow, withPageRank, withUrlLen, withTitleLen, withBodyLen, withHeaderLen, withAnchorLen, withUrlPDF);
    try{
      model = new LibSVM();
     //0. Build the attributes
   	 /* Build attributes list */
  	
  	 attributes.add(new Attribute("url_w"));
  	 attributes.add(new Attribute("title_w"));
  	 attributes.add(new Attribute("body_w"));
  	 attributes.add(new Attribute("header_w"));
  	 attributes.add(new Attribute("anchor_w"));
  	 
  	if(withBM25)
  		attributes.add(new Attribute("bm25_w"));
  	if(withSmallestWindow)
  		attributes.add(new Attribute("smallWindow_w"));
  	if(withPageRank)
  		attributes.add(new Attribute("pageRank_w"));
  	if(withUrlLen)
  		attributes.add(new Attribute("UrlLen"));
  	if(withTitleLen)
  		attributes.add(new Attribute("TitleLen"));
  	if(withBodyLen)
  		attributes.add(new Attribute("BodyLen"));
  	if(withHeaderLen)
  		attributes.add(new Attribute("HeaderLen"));
  	if(withAnchorLen)
  		attributes.add(new Attribute("AnchorLen"));  	
  	if(withUrlPDF)
  		attributes.add(new Attribute("UrlPDF"));
  	

    } catch (Exception e){
      e.printStackTrace();
    }
    
     
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
    
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel,  boolean withBM25, boolean withSmallestWindow, boolean withPageRank, boolean withUrlLen, boolean withTitleLen, boolean withBodyLen, boolean withHeaderLen, boolean withAnchorLen, boolean withUrlPDF){
		super(withBM25, withSmallestWindow, withPageRank, withUrlLen, withTitleLen, withBodyLen, withHeaderLen, withAnchorLen, withUrlPDF);
    try{
      model = new LibSVM();
      //0. Build the attributes
      /* Build attributes list */
   	
   	  attributes.add(new Attribute("url_w"));
   	  attributes.add(new Attribute("title_w"));
   	  attributes.add(new Attribute("body_w"));
   	  attributes.add(new Attribute("header_w"));
   	  attributes.add(new Attribute("anchor_w"));
  	  if(withBM25)
  		attributes.add(new Attribute("bm25_w"));
  	  if(withSmallestWindow)
  		attributes.add(new Attribute("smallWindow_w"));
  	  if(withPageRank)
  		attributes.add(new Attribute("pageRank_w"));
  	if(withUrlLen)
  		attributes.add(new Attribute("UrlLen"));
  	if(withTitleLen)
  		attributes.add(new Attribute("TitleLen"));
  	if(withBodyLen)
  		attributes.add(new Attribute("BodyLen"));
  	if(withHeaderLen)
  		attributes.add(new Attribute("HeaderLen"));
  	if(withAnchorLen)
  		attributes.add(new Attribute("AnchorLen"));  
  	
  	  if(withUrlPDF)
  		attributes.add(new Attribute("UrlPDF"));

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
  
  // overloaded to extract testing features
  public Pair<Instances, Map <String, List<Pair<String, Integer>>>>  extractPointWiseFeatures(String test_data_file,
			 Map<String, Double> idfs) throws Exception{
	  
	  Pair<Instances, Map <String, List<Pair<String, Integer>>>> featuresANDindices = extractPointWiseFeatures(test_data_file,
				"", idfs); 	
	  return featuresANDindices; 
}
  public Pair<Instances, Map <String, List<Pair<String, Integer>>>>  extractPointWiseFeatures(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception{
	  	  
	  	Pair<Instances, Map <String, List<Pair<String, Integer>>>> featuresANDindices = null; 
	  
	  	Instances pointWiseFeatures = null;
	  	Map <String, List<Pair<String, Integer>>> indexMap = new HashMap<String, List<Pair<String, Integer>>>(); 
		
	  	ArrayList<Attribute> point_attributes = new ArrayList<Attribute>(attributes);
	  	point_attributes.add(new Attribute("relevance_score"));
		pointWiseFeatures = new Instances("pre_train_dataset", point_attributes, 0);
				
		// load the training data
		Map<Query,List<Document>> queryDict = Util.loadTrainData(train_data_file);
		Map<String, Map<String, Double>> relevanceDict = null ;
		//  load the relevance score data (if the filename is provided)
		if(!train_rel_file.equals(""))
			relevanceDict = Util.loadRelData(train_rel_file); 
		// create a scorer to extract features
		AScorer extractor = new tf_idfExtractor(idfs);
		AScorer bm25Scorer = null; 
		AScorer smallWindowScorer = null; 
		if(withBM25)		
			bm25Scorer = new BM25Scorer(idfs, queryDict);					
		if(withSmallestWindow)		
			smallWindowScorer = new SmallestWindowScorer(idfs, queryDict);
		boolean withMoreFeatures = (withUrlLen || withTitleLen|| withBodyLen|| withHeaderLen|| withAnchorLen|| withUrlPDF); 	
		Map<String, Double> tf_idfs, bm25, smallestWindow, moreFeatures; 
		
		Map<String, Double> queryRelMap = null;
		// go over every (query, document) instance in the training data, compute the features, and add it to the data set	
		for (Query query : queryDict.keySet()) {
			if(!train_rel_file.equals(""))
				queryRelMap = relevanceDict.get(query.query);	
			List<Pair<String, Integer>> queryDocFeatureList = new ArrayList<Pair<String, Integer>>();  
			// Loop through the documents for query, getting scores
			for (Document doc : queryDict.get(query)) {
				// for this query document pair
				Instance inst = new DenseInstance(1+attributes.size());						
			    inst.setDataset(pointWiseFeatures);				
				// 1. get the tf-idf features for every one of the five fields 
				tf_idfs = extractor.getFeatures(doc, query);				
				int fieldInd = 0; 
				for(String field: tf_idfs.keySet())
					inst.setValue(fieldInd++, tf_idfs.get(field)); 
				if(withBM25)
				{
					bm25 = bm25Scorer.getFeatures(doc, query); 
					inst.setValue(fieldInd++, bm25.get("bm25")); 
				} 
				if(withSmallestWindow)
				{ 
					smallestWindow = smallWindowScorer.getFeatures(doc, query); 
					inst.setValue(fieldInd++, smallestWindow.get("smallWindow")); 
				}
				if(withPageRank)
					inst.setValue(fieldInd++, (double)doc.page_rank);
				if(withMoreFeatures)  
				{ 
					moreFeatures = extractor.getMoreFeatures(doc, query);
					if(withUrlLen)
						inst.setValue(fieldInd++, moreFeatures.get("urlLen"));
					if(withTitleLen)
						inst.setValue(fieldInd++, moreFeatures.get("titleLen"));
					if(withBodyLen)
						inst.setValue(fieldInd++, moreFeatures.get("bodyLen"));
					if(withHeaderLen)
						inst.setValue(fieldInd++, moreFeatures.get("headerLen"));
					if(withAnchorLen)
						inst.setValue(fieldInd++, moreFeatures.get("anchorLen"));					
					if(withUrlPDF)
						inst.setValue(fieldInd++, moreFeatures.get("urlPDF"));
				} 
				// 2. Get the relevance score
				inst.setValue(fieldInd, 1.0); // leave it as 1 for test set, it does not matter
				if(!train_rel_file.equals(""))
					inst.setValue(fieldInd, queryRelMap.get(doc.url));
				
				// 3. add the data instance for this query document pair to the data set								
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
				 extractPointWiseFeatures(train_data_file, train_rel_file, idfs);
		Instances pointWiseFeatures 					   = featuresANDindices.getFirst(); 
		Map <String, List<Pair<String, Integer>>> indexMap = featuresANDindices.getSecond();  
		
		// 2. Standardize the pointwise features
		Instances standardizedFeatures = standardizeFeatures(pointWiseFeatures); 
		
		// 3. Get the pairwise features 
		Instances pairWiseDataset = null;
		ArrayList<Attribute> pair_attributes = new ArrayList<Attribute>(attributes); 
		pair_attributes.add(new Attribute("class", Arrays.asList(classLabels)));
		pairWiseDataset = new Instances("train_dataset", pair_attributes, 0);
		pairWiseDataset.setClassIndex(pairWiseDataset.numAttributes() - 1);
		
	/*	System.out.println("class attribute" + " " + pairWiseDataset.classAttribute().value(0));
	    System.out.println("class attribute" + " " + pairWiseDataset.classAttribute().value(1));*/
	    
	//	System.out.println("number of attributes is " + pairWiseDataset.numAttributes()); 
		List<Pair<String, Integer>> queryDocFeatureList;
		int queryDocInd1 = 0, queryDocInd2 = 0; 
		double relevance1 = 0.0, relevance2 = 0.0;  
		int numPositive = 0, numNegative = 0;
		int label = 0;
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
		
						double[] instance = new double[attributes.size()];
						label = (int)Math.signum(relevance1 - relevance2);
						
						//balance the features: negative and positive instances must be balanced
						if(label*(numPositive - numNegative) >0) // label > 0 and numPositive > numNegative OR label < 0 and numPositive < numNegative 
							flip = -1; 
						else 
							flip = 1; 
						for(int attInd = 0; attInd < attributes.size(); attInd++)
							instance[attInd] = flip * ( standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute(attInd))
												 - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute(attInd)));
						label = (int)flip*label;
						String classLabel = Integer.toString(label);
						
						Instance inst = new DenseInstance(1+attributes.size());						
					    inst.setDataset(pairWiseDataset);
					    int ind = 0;
					    for(ind = 0 ; ind < attributes.size(); ind++ )
					    	inst.setValue(ind, instance[ind]);					    				     
					    inst.setValue(ind, classLabel); 
					    
						pairWiseDataset.add(inst);
						
						if(label>0)
							numPositive++; 
						else 
							numNegative++;
						
					}
				} 
		} 
		
		// check if we balanced the features: negative and positive instances must be balanced
		/*
		System.out.println("numPositive = " + numPositive);
		System.out.println("numNegative = " + numNegative); 
		*/
	/*	//4. convert last attribute, the target/label, to categorical as we have classification not regression 
		NumericToNominal convert= new NumericToNominal();
	    String[] options= new String[2];
	    options[0]="-R";
	    options[1]="6";  //range of variables to make nominal: just the last one
        convert.setOptions(options);
	    convert.setInputFormat(pairWiseDataset);
	    Instances finalPairWiseDataset=Filter.useFilter(pairWiseDataset, convert);	    
	    finalPairWiseDataset.setClassIndex(pairWiseDataset.numAttributes() - 1);*/
	    
	    /*System.out.println(" category " + finalPairWiseDataset.get(0).stringValue(finalPairWiseDataset.attribute("relevance_score"))); 
	    System.out.println(" category " + finalPairWiseDataset.get(1).stringValue(finalPairWiseDataset.attribute("relevance_score")));*/
		return pairWiseDataset;
		
	}

	@Override
	public Classifier training(Instances dataset) throws Exception {
		/*
		 * @TODO: Your code here
		 */		
		model.buildClassifier(dataset);
		return model;
	}

	
	
	// implement pairwise test features extraction, will work for bothlinear and nonlinear SVM  
	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) throws Exception {
		/*
		 * @TODO: Your code here
		 */
		// 1. Get the pointwise features  
		Pair<Instances, Map <String, List<Pair<String, Integer>>>>  featuresANDindices = 
				 extractPointWiseFeatures(test_data_file, idfs);
		Instances pointWiseFeatures 					   = featuresANDindices.getFirst(); 
		Map <String, List<Pair<String, Integer>>> indexMap = featuresANDindices.getSecond();  
		
		// 2. Standardize the pointwise features
		Instances standardizedFeatures = standardizeFeatures(pointWiseFeatures); 
		
		// 3. Get the pairwise features 
		TestFeatures myTestFeatures = new TestFeatures(); 	
		ArrayList<Attribute> pair_attributes = new ArrayList<Attribute>(attributes); 
		pair_attributes.add(new Attribute("class", Arrays.asList(classLabels)));		
		myTestFeatures .features = new Instances("test_dataset", pair_attributes, 0);
		myTestFeatures.index_map = new HashMap<String, Map<String, Integer>>(); 
		
	//	System.out.println("number of attributes is " + pairWiseDataset.numAttributes()); 
		List<Pair<String, Integer>> queryDocFeatureList;
		int queryDocInd1 = 0, queryDocInd2 = 0; 
		String twoURLSConcat = ""; 
		for(String q: indexMap.keySet())
		{ 			
			queryDocFeatureList = indexMap.get(q);
			Map<String, Integer> queryDocFeatureMap = new HashMap<String, Integer>(); 
			for(int ind1 = 0; ind1 < queryDocFeatureList.size(); ind1++)
				for( int ind2 = ind1+1; ind2 < queryDocFeatureList.size(); ind2++)
				{ 
					queryDocInd1 = queryDocFeatureList.get(ind1).getSecond();
					queryDocInd2 = queryDocFeatureList.get(ind2).getSecond(); 
									
					double[] instance = new double[attributes.size()];
					
					for(int attInd = 0; attInd < attributes.size(); attInd++)				
						instance[attInd] = standardizedFeatures.instance(queryDocInd1).value(standardizedFeatures.attribute(attInd))
					        	       	 - standardizedFeatures.instance(queryDocInd2).value(standardizedFeatures.attribute(attInd));
					
					String classLabel = "1"; // does not matter
					
					Instance inst = new DenseInstance(1+attributes.size());						
				    inst.setDataset(myTestFeatures.features);
				    int ind = 0;
				    for(ind = 0 ; ind < attributes.size(); ind++ )
				    	inst.setValue(ind, instance[ind]);					    				     
				    inst.setValue(ind, classLabel); // does not matter 
				    
					myTestFeatures.features.add(inst);
					twoURLSConcat = queryDocFeatureList.get(ind1).getFirst() + " " +  queryDocFeatureList.get(ind2).getFirst(); 
					queryDocFeatureMap.put(twoURLSConcat, myTestFeatures.features.size()-1); 
					
				} 
			myTestFeatures.index_map.put(q, queryDocFeatureMap); 
		} 
		
		
		return myTestFeatures;
	}
	
	
	@Override
	// will implement on pairwise features and use classifyInstance 
	// that outputs a binary: the class/category of the pair. 
	public Map<String, List<String>> testing(TestFeatures tf,
			final Classifier model) throws Exception {
		/*
		 * @TODO: Your code here
		 */		
		
		Map<String,List<String>> ranked_queries = new HashMap<String, List<String>>(); 		
		final Instances test_dataset = tf.features; /* The dataset you built in Step 3 */
		test_dataset.setClassIndex(test_dataset.numAttributes()-1);
	/*	System.out.println("class attribute" + " " + test_dataset.classAttribute().value(0));
	    System.out.println("class attribute" + " " + test_dataset.classAttribute().value(1));*/

		// Go over every query in the test features
		int queryDocInd = 0; 
		 
		for(String q: tf.index_map.keySet())
		{ 
			// 1. For every query, unfold the url pairs and put in url set of unique urls for this query
		//	System.out.println("testing query: " + q); 
			Set<String> urlSet = new LinkedHashSet<String>();
			final Map<String, Integer> queryDocFeatureMap = tf.index_map.get(q);
			for(String twoURLSConcat: queryDocFeatureMap.keySet())
			{ 
				String urlArray[] = twoURLSConcat.split(" "); 
				urlSet.add(urlArray[0]); 
				urlSet.add(urlArray[1]);
			}
			//2. turn the url set to a list
			List<String> rankedUrl = new ArrayList<String>(urlSet);
			
		//	System.out.print("\n"); 
			// 3. Sort the urls based on classifier output that compares each pair			
			Collections.sort(rankedUrl, new Comparator<String>() {
			@Override
			public int compare(String o1, String o2) {
				/*
				 * @//TODO : Your code here
				 */
				String twoURLSConcat1 = o1 + " " + o2;
				String twoURLSConcat2 = o2 + " " + o1;
				int queryDocInd = 0;
				int check = 0; 
				int flip = 1; 
				if(queryDocFeatureMap.containsKey(twoURLSConcat1))
				{ 
					check++; 
					queryDocInd = queryDocFeatureMap.get(twoURLSConcat1);
				} 
				if(queryDocFeatureMap.containsKey(twoURLSConcat2))
				{ 
					check++; 
					queryDocInd = queryDocFeatureMap.get(twoURLSConcat2);
					flip = -1; 
				}
									
				// sanity check for debugging
				if(check==0)
					System.out.println("none found!!"); 
				if(check==2)
					System.out.println("both found");
				
				int classIndex = 0;
				try {
					 
					classIndex = (int)model.classifyInstance(test_dataset.instance(queryDocInd));
					 
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				 				
				classIndex = (classIndex + (1-flip)/2)%2; // if flip =1, ie no flip, keep classindex as it is, 
														  // if flip = -1, 0 -- > 1 and 1 --> 0 
				String classLabel = test_dataset.classAttribute().value(classIndex); 
				
				int comparisonResult = -Integer.parseInt(classLabel); // we want to sort in decreasing order
				
				return comparisonResult; 
				
			}	
			});
		
				
			ranked_queries.put(q, rankedUrl); 
		} 
		
		return ranked_queries;
	}
	
	
	//// ----------------CODE I AM NO MORE USING, WORKS only for linear SVM, as it is pointwise
	
	// works only for linear svm
			// @Override
			public TestFeatures pointWiseExtract_test_features(String test_data_file,
					Map<String, Double> idfs) throws Exception {
				/*
				 * @TODO: Your code here
				 */
				// 1. Get the pointwise features  
				Pair<Instances, Map <String, List<Pair<String, Integer>>>>  featuresANDindices = 
						 extractPointWiseFeatures(test_data_file, idfs);
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
		
		// Be careful as we are getting this warning message: 
		// LibSVM get coefficients: note that the sign might be negated. It's best to use the classifyInstance method.
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
			score += weights[weightInd];  // not necessary, it is a constant
			
			return score; 
		}
		//@Override
		// pointWiseTesting: works for libSVM, by extracting the weights using coefficients and computing the dot product with the pointwise features
		// in this case classifyInstance does not work as it is output is binary and is actually the class/category. 
		public Map<String, List<String>> pointWiseTesting(TestFeatures tf,
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
