package cs276.pa4;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.pa4.AScorer;
import cs276.pa4.Document;
import cs276.pa4.Query;


public class tf_idfExtractor extends AScorer 
{
	double smoothingBodyLength = 500.0; // Smoothing factor when the body length is 0.
	public tf_idfExtractor (Map<String,Double> idfs) 
	{
		super(idfs);
	}
	
	// Normalize the term frequencies. Note that we should give uniform normalization to all fields as discussed
		// in the assignment handout.
		// also add sublinear scaling here if needed 
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {

		/*
		 * @//TODO : Your code here
		 */
		double bodyLength_inv;
		bodyLength_inv= 1.0/((double)d.body_length + smoothingBodyLength);				 			
		double freq; 
		
		for (String term : tfs.get("url").keySet())	
		{		
			freq = tfs.get("url").get(term); 
			if(freq!=0)

			{ 				
				freq *= bodyLength_inv;					
				tfs.get("url").put(term, freq);
			} 
			
			freq = tfs.get("title").get(term); 
			if(freq!=0)
			{ 						
				freq *= bodyLength_inv;
				tfs.get("title").put(term, freq);
			}
			
			freq = tfs.get("body").get(term); 
			if(freq!=0)

			{
				freq *= bodyLength_inv;
				tfs.get("body").put(term, freq);
			}
			
			freq = tfs.get("header").get(term); 
			if(freq!=0)

			{ 				
				freq *= bodyLength_inv;
				tfs.get("header").put(term, freq);
			}
			
			freq = tfs.get("anchor").get(term); 
			if(freq!=0)

			{ 				
				freq *= bodyLength_inv;
				tfs.get("anchor").put(term, freq);
			}
		}
	}

	@Override
	public Map<String, Double> getFeatures(Document d, Query q) {
		
		
		
		Map<String, Double> tf_idf = new HashMap<String, Double>();
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = this.getQueryFreqs(q);
		double val = 0.0; 
		for(String field: tfs.keySet())
		{ 		
			val = 0.0;   
			for(String term: tfs.get(field).keySet())
			{ 
				val += tfs.get(field).get(term) * tfQuery.get(term); 
			}
			tf_idf.put(field, val); 
		}
			
		return tf_idf; 
	}
}
