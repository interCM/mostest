% required input
if ~exist('out', 'var'), error('out file prefix is required'); end
if ~exist('mat_name', 'var'), error('mat file is required'); end                % path to mat file produced by mostest.py

% optional arguments
if ~exist('use_paretotails', 'var'), use_paretotails = false; end;              % use paretotails instead of the gamma and beta functions to fit the distribution of the MOSTest & minP test statistic under null
if ~exist('paretotails_quantile', 'var'), paretotails_quantile = 0.99; end;     % a number close to 1.0, used as a second argument in MATLAB's paretotails

% =============== end of parameters section =============== 

fprintf('loading %s... ', mat_name);
load(mat_name);
fprintf('OK.\n')
npheno=size(C0, 1);


maxlogpvecs = -log10(minpvecs);
ivec_snp_good = all(isfinite(mostvecs+minpvecs+maxlogpvecs));

[hc_maxlogpvecs hv_maxlogpvecs] = hist(maxlogpvecs(2,ivec_snp_good),1000); chc_maxlogpvecs = cumsum(hc_maxlogpvecs)/sum(hc_maxlogpvecs);
[hc_mostvecs hv_mostvecs] = hist(mostvecs(2,ivec_snp_good),1000); chc_mostvecs = cumsum(hc_mostvecs)/sum(hc_mostvecs);

if use_paretotails
  pd_minpvecs = paretotails(minpvecs(2,ivec_snp_good), 0.00, paretotails_quantile);
  pd_minpvecs_params = upperparams(pd_minpvecs);
  pd_mostvecs = paretotails(mostvecs(2,ivec_snp_good), 0.00, paretotails_quantile);
  pd_mostvecs_params = upperparams(pd_mostvecs);
else
  pd_minpvecs = fitdist(colvec(minpvecs(2,ivec_snp_good)),'beta'); % Not a great fit
  pd_minpvecs_params = [pd_minpvecs.a, pd_minpvecs.b];
  pd_mostvecs = fitdist(colvec(mostvecs(2,ivec_snp_good)),'gamma'); % Seems to work -- beta and wbl  do not
  pd_mostvecs_params = [pd_mostvecs.a, pd_mostvecs.b];
end

cdf_minpvecs=cdf(pd_minpvecs,10.^-hv_maxlogpvecs,'upper');
cdf_mostvecs = pd_mostvecs.cdf(hv_mostvecs);

maxlogpvecs_corr = -log10(cdf(pd_minpvecs,minpvecs));
mostvecs_corr = -log10(cdf(pd_mostvecs,mostvecs,'upper'));
fprintf('Done.\n')

fprintf('GWAS yield minP: %d; MOST: %d\n',sum(maxlogpvecs_corr(1,ivec_snp_good)>-log10(5e-8)),sum(mostvecs_corr(1,ivec_snp_good)>-log10(5e-8)));
fprintf('%i\t%.2f\t%.3f\t%.3f\t%.3f\t%.3f\t\n', npheno, cond(C0), pd_minpvecs_params(1), pd_minpvecs_params(2), pd_mostvecs_params(1), pd_mostvecs_params(2)) 

most_time_sec = toc;

minp_log10pval_orig = maxlogpvecs_corr(1, :);
most_log10pval_orig = mostvecs_corr(1, :);
minp_log10pval_perm = maxlogpvecs_corr(2, :);
most_log10pval_perm = mostvecs_corr(2, :);
fname=sprintf('%s.mat', out);
hv_logpdfvecs=hv_mostvecs; cdf_logpdfvecs=cdf_mostvecs; chc_logpdfvecs=chc_mostvecs;
fprintf('saving %s... ', fname);
save(fname, '-v7', ...
 'most_log10pval_orig', 'minp_log10pval_orig', ...
 'most_log10pval_perm', 'minp_log10pval_perm', ...
 'nvec', 'freqvec', 'ivec_snp_good', ...
 'measures', 'ymat_corr', 'C0', 'C1', ...
 'minpvecs', 'mostvecs', ...
 'hv_maxlogpvecs', 'hc_maxlogpvecs', 'chc_maxlogpvecs', 'cdf_minpvecs', ...
 'hv_mostvecs', 'hc_mostvecs', 'chc_mostvecs', 'cdf_mostvecs', ...
 'pd_minpvecs_params', 'pd_mostvecs_params', 'gwas_time_sec', 'most_time_sec');
fprintf('Done.\n')

fprintf('MOSTest analysis is completed.\n')