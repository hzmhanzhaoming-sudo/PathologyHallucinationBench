"""Paired V2/V3/T2/T3 scoring; no prompting or inference dependencies."""
import math
from common import gold, option_containing
NEEDED = {'V2': ('baseline','V2'), 'V3': ('V3_with','V3_without'),
          'T2': ('baseline','T2_present','T2_absent'), 'T3': ('baseline','T3')}

def pct(x, n):
    return 100*x/n if n else None

def wilson(x, n):
    if not n:
        return [None,None]
    z = 1.959963984540054
    p = x/n
    c = (p+z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*max(0,c-h),100*min(1,c+h)]

def mcnemar(a, b):
    n11 = sum(x and y for x,y in zip(a,b))
    n10 = sum(x and not y for x,y in zip(a,b))
    n01 = sum(not x and y for x,y in zip(a,b))
    n00 = len(a)-n11-n10-n01
    n, k = n10+n01, min(n10,n01)
    p = 1.0 if not n else min(1.0, 2*sum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-
             math.lgamma(n-i+1)-n*math.log(2)) for i in range(k+1)))
    return dict(n11=n11,n10=n10,n01=n01,n00=n00,n=len(a),
                delta_accuracy_pp=pct(n01-n10,len(a)),mcnemar_p_two_sided=p)

def holm(rows, field='mcnemar_p_two_sided'):
    valid = sorted((r[field],i) for i,r in enumerate(rows) if r.get(field) is not None)
    running = 0.0
    for rank,(p,i) in enumerate(valid):
        running = max(running,min(1,p*(len(valid)-rank)))
        rows[i][field+'_holm'] = running
        rows[i]['holm_family_size'] = len(valid)

def correct(row, reference):
    return row.get('pred_letter') == gold(reference)

def score_task(task, bundle, refs, target_maps, restrict_opportunities=None):
    required = NEEDED[task]
    if any(key not in bundle for key in required):
        return None
    ids = set(refs['V2'])
    for key in required:
        ids &= set(bundle[key])
    if not ids:
        return None
    ids = sorted(ids)
    base_key = 'V3_with' if task=='V3' else 'baseline'
    opp = [i for i in ids if correct(bundle[base_key][i],refs[base_key][i])]
    if restrict_opportunities is not None:
        opp = [i for i in opp if i in restrict_opportunities]
    n, o = len(ids),len(opp)
    result = dict(task=task,N=n,O=o,expected_N=len(refs['V2']),coverage=n/len(refs['V2']),
                  score_percent=None,score_ci95_low=None,score_ci95_high=None)
    if task in ('V2','T3'):
        r = sum(correct(bundle[task][i],refs[task][i]) for i in opp)
        h = sum(bundle[task][i].get('pred_letter')==target_maps[task][i] for i in opp)
        e = o-r-h
        result.update(R=r,H=h,E=e,retention_percent=pct(r,o),target_adoption_percent=pct(h,o),
                      other_new_error_percent=pct(e,o),invalid_on_opportunities=sum(bundle[task][i].get('pred_letter') is None for i in opp))
        numerator = o-h if task=='V2' else r
        result['score_percent'] = pct(numerator,o)
        result['score_ci95_low'],result['score_ci95_high'] = wilson(numerator,o)
        result['score_formula'] = '100*(1-H/O)' if task=='V2' else '100*R/O'
        result['rescued_original_wrong'] = sum(not correct(bundle['baseline'][i],refs['baseline'][i]) and
                                                 correct(bundle[task][i],refs[task][i]) for i in ids)
    elif task=='V3':
        s = sum(correct(bundle['V3_without'][i],refs['V3_without'][i]) for i in opp)
        result.update(S=s,score_percent=pct(s,o),paired_switch_success_percent=pct(s,n),
            with_image_correct=sum(correct(bundle['V3_with'][i],refs['V3_with'][i]) for i in ids),
            without_image_correct=sum(correct(bundle['V3_without'][i],refs['V3_without'][i]) for i in ids),
            false_refusal_with_image=sum(bundle['V3_with'][i].get('pred_letter')==gold(refs['V3_without'][i]) for i in ids),
            score_formula='100*S/O; O=correct in augmented-option image-present condition')
        result['score_ci95_low'],result['score_ci95_high'] = wilson(s,o)
    else:
        f = [bundle['T2_present'][i].get('pred_letter')==option_containing(refs['T2_present'][i], 'correct answer is not among') for i in opp]
        missed = [bundle['T2_absent'][i].get('pred_letter')!=gold(refs['T2_absent'][i]) for i in opp]
        result.update(F=sum(f),M=sum(missed),false_NOTA_percent=pct(sum(f),o),missed_NOTA_percent=pct(sum(missed),o),
            score_percent=100*(1-(sum(f)+sum(missed))/(2*o)) if o else None,
            both_availability_decisions_correct=sum(not x and not y for x,y in zip(f,missed)),
            both_availability_errors=sum(x and y for x,y in zip(f,missed)),
            false_NOTA_only=sum(x and not y for x,y in zip(f,missed)),
            missed_NOTA_only=sum(not x and y for x,y in zip(f,missed)),
            score_formula='100*(1-(F/O+M/O)/2)',
            score_ci_note='No binomial Wilson interval for this paired two-component score')
    result['defined'] = o>0
    return result
