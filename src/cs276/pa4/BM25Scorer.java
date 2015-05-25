package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Skeleton code for the implementation of a BM25 Scorer in Task 2.
 */
public class BM25Scorer extends AScorer {
	
	private static boolean debug = false;
	
	Map<Query,List<Document>> queryDict; // query -> url -> document

	public BM25Scorer(Map<String,Double> idfs, Map<Query,List<Document>> queryDict) {
		super(idfs);
		this.queryDict = queryDict;
		this.calcAverageLengths();
	}


	/////////////// Weights /////////////////
	static public Map<String, Double> weights;
	static {
		weights = new HashMap<>();
		weights.put("url", 1.0);
		weights.put("title", 0.8);
		weights.put("body", 0.4);
		weights.put("header", 0.6);
		weights.put("anchor", 0.2);
	}
/*
	double urlweight = -1;
	double titleweight  = -1;
	double bodyweight = -1;
	double headerweight = -1;
	double anchorweight = -1;
*/
	/////// BM25 specific weights ///////////
	// title.url.anchor.body.header
	static public Map<String, Double> B;
	static {
		B = new HashMap<>();
		B.put("url", 0.8);
		B.put("title", 1.0);
		B.put("body", 0.4);
		B.put("header", 0.2);
		B.put("anchor", 0.6);
	}
/*	
	double burl=-1;
	double btitle=-1;
	double bheader=-1;
	double bbody=-1;
	double banchor=-1;
*/
	static String[] PARAMS = {"k1","pageRankLambda","pageRankLambdaPrime"};

	public static double k1 = 0.5;
	public static double pageRankLambda = 0.5;
	public static double pageRankLambdaPrime = 0.5;
	//////////////////////////////////////////

	/////// BM25 data structures - feel free to modify ///////

	Map<Document,Map<String,Double>> lengths; // Document -> field -> length
	Map<String,Double> avgLengths;  // field name -> average length
	Map<Document,Double> pagerankScores; // Document -> pagerank score

	//////////////////////////////////////////
	
	public static Set<String> paramPermutations()
	{
		Set<String> perms = new HashSet<>();
		
		for (int i=0; i<PARAMS.length; i++)
		{
			for (int j=0; j<PARAMS.length; j++)
			{
				if (j == i) continue;
				for (int k=0; k<PARAMS.length; k++)
				{
					if (k == j || k == i) continue;
					perms.add(TFTYPES[i] + "." + TFTYPES[j] + "." + TFTYPES[k]);
				}

			}
		}
		
		return perms;
	}


	// Set up average lengths for bm25, also handles pagerank
	public void calcAverageLengths() {
		lengths = new HashMap<Document,Map<String,Double>>();
		avgLengths = new HashMap<String,Double>();
		pagerankScores = new HashMap<Document,Double>();
		
		/*
		 * @//TODO : Your code here
		 */
		
		for (String tftype : TFTYPES) { avgLengths.put(tftype, 0.0); }
		
		int docCount = 0;
		int length = 0;
		
		for (Query query : queryDict.keySet())
		{
			for (Document d : queryDict.get(query))
			{
				String url = d.url;
				
				HashMap<String, Double> docLengths = new HashMap<>();

				docCount++;
				
				if (debug) System.out.println(d);
				
				// "url","title","body","header","anchor"
				length = getURLTerms(d).length;
				docLengths.put("url", (double)length);
				avgLengths.put("url", avgLengths.get("url") + length);
				
				length = getTitleTerms(d).length;
				docLengths.put("title", (double)length);
				avgLengths.put("title", avgLengths.get("title") + length);
				
				docLengths.put("body",  0.0);
				if (null != d.body_hits)
				{
					for (String term : d.body_hits.keySet())
					{
						length = d.body_hits.get(term).size();
						docLengths.put("body", docLengths.get("body") + length);
						avgLengths.put("body", avgLengths.get("body") + length);
					}
				}
				
				docLengths.put("header", 0.0);
				if (null != d.headers)
				{
					for (String header : d.headers)
					{
						length = getHeaderTerms(header).length;
						docLengths.put("header", docLengths.get("header") + length);
						avgLengths.put("header", avgLengths.get("header") + length);
					}
				}
				
				docLengths.put("anchor",  0.0);
				if (null != d.anchors)
				{
					for (String anchor : d.anchors.keySet())
					{
						length = getAnchorTerms(anchor).length * d.anchors.get(anchor);
						docLengths.put("anchor", docLengths.get("anchor") + length);
						avgLengths.put("anchor", avgLengths.get("anchor") + length);
					}
				}
				
				lengths.put(d, docLengths);
				pagerankScores.put(d, (double)d.page_rank);
			}
		}
		
		//normalize avgLengths
		for (String tftype : this.TFTYPES) {
			/*
			 * @//TODO : Your code here
			 */
			avgLengths.put(tftype, avgLengths.get(tftype) / (double)docCount);
		}

	}

	////////////////////////////////////


	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d) {
		double score = 0.0;
		
		/*
		 * @//TODO : Your code here
		 */
		for (String term : q.words)
		{
			double w_dt = 0.0;
			for (String tftype : TFTYPES) 
			{
				w_dt += weights.get(tftype) * tfs.get(tftype).get(term);
			}
			double idf_t = (null == idfs.get(term)) ? idfs.get(IDF_MAX) : idfs.get(term);
			
			score += ( w_dt / (k1 + w_dt) ) * idf_t;
		}
		score += pageRankLambda * V_j(pagerankScores.get(d));
		
		return score;
	}
	
	private double V_j(Double f)
	{
		double score = 0.0;
		// Choose log/saturation/sigmoid function
		
		// log
		score = Math.log(pageRankLambdaPrime + f);
		
		// saturation
		//score = f / (pageRankLambdaPrime + f);
		
		// sigmoid - do not have second derivative
		//score = 1 / (1 + Math.exp(-f * pageRankLambdaPrime));
		
		return score;
	}

	//do bm25 normalization
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
		/*
		 * @//TODO : Your code here
		 */
		if (debug) System.out.println(tfs);
		
		for (String tftype : TFTYPES) 
		{ 
			for (String term : q.words)
			{
				double tf_dft  = tfs.get(tftype).get(term);
				double len_df  = lengths.get(d).get(tftype);
				double avlen_f = avgLengths.get(tftype);
				double B_f = B.get(tftype);
				
				double ftf_dft = avlen_f == 0.0 ? 0.0 : tf_dft / ( 1 + B_f * ( (len_df / avlen_f) - 1 ));
				
				tfs.get(tftype).put(term, ftf_dft);
			}
		}
		
		if (debug) System.out.println(tfs);
	}


	@Override
	public double getSimScore(Document d, Query q) {
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);

	    return getNetScore(tfs,q,tfQuery,d);
	}
	
}
