function [out] = rssq_fun(x)
out = squeeze(sum(abs(x).^2,3)).^(1/2);
end

