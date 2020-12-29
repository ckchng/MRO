
% make relative rotations from GT
sampleIdx = 1:1:450;
ORB_pose = load('ORB_pose');

CvORB_sampled = ORB_pose.CvORB(:,:,sampleIdx);

rotsGT = cell(1,length(CvORB_sampled));
k = 1;
for i=sampleIdx
    rotsGT{k} = ORB_pose.CvORB(:,:,i)';
    k = k+1;
end

% make block rotation matrix
m = size(rotsGT,2);
R = cell(m,m);
for i=1:m
    Ri = rotsGT{i};
    R{i,i} = zeros(3,3);
    for j=(i+1):m
        Rj = rotsGT{j};
        R{i,j} = Ri'*Rj;
        %R{i,j} = Rj*Ri';
        
        R{j,i} = R{i,j}';
    end
end

% run rot avg
rotsGT = rotsGT(1:m);
rots_init = cellfun(@(x) roterr(.1*pi/180)*x, rotsGT, 'UniformOutput', 0);

[ravg, iter, objVals] = rotavg(rots_init, cell2mat(R));
%[ravg] = rotavg_l1irls(R, rots_init);


avgErr = zeros(1,m);
initErr = zeros(1,m);
for i=1:m
    avgErr(i)  = rotdist(ravg{i}, rotsGT{i});
    initErr(i) = rotdist(rots_init{i}, rotsGT{i});
end

figure, plot(avgErr.*180/pi, 'linewidth', 2)
hold on
plot(initErr.*180/pi)
legend('avg', 'init ')
