function [ theta ] = rotdist( S, R)
%Angular distance between two rotation matrices. 

n = norm(S-R,'fro');
theta = 2*asin(n/(2*sqrt(2)));
end

