function [E] = solve_l1l2(W,lambda)

n = size(W,2); 
E = W; 
for i=1:n
    E(:,i) = solve_l1(W(:,i),lambda); 
end
end

function [x] = solve_l2(w,lambda)

nw = norm(w); 
if nw>lambda
    x = (nw-lambda)*w/nw; 
else
    x = zeros(length(w),1);
end
end

function [x] = solve_l1(w, lambda)

x = sign(w) .* max(abs(w) - lambda, 0); 
end