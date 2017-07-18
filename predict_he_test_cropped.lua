require 'paths'
require 'lfs'
paths.dofile('util.lua')
paths.dofile('img.lua')

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

m = torch.load('umich-stacked-hourglass.t7')   -- Load pre-trained model
print("Model loaded.")


filelist = {}
i=0

for file in lfs.dir[[/home/emredog/git/pose-hg-demo/images/he/test_256_cropped/]] do
    if lfs.attributes(file,"mode") ~= "directory" then 
        filelist[i] = file
        i = i+1
    end
end

nbSamples = i
print('Found ' .. nbSamples .. ' images.')

-- Displays a convenient progress bar
xlua.progress(0,nbSamples)

-- Placeholder for predictions
preds = torch.Tensor(nbSamples,16,2)
paths = {} 

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nbSamples do
    -- Set up input image
    -- im0020_C1_256.jpg
    impath = 'images/he/test_256_cropped/' .. filelist[i-1]
    local im = image.load(impath)
    local center = {128, 128}
    local scale = 1.0
    -- local inp = crop(im, center, scale, 0, 256)
    inp = im

    -- Get network output
    local out = m:forward(inp:view(1,3,256,256):cuda())
    cutorch.synchronize()
    local hm = out[#out][1]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img = getPreds(hm, center, scale)

    preds[i]:copy(preds_img)
    paths[i-1] = impath

    xlua.progress(i,nbSamples)

    -- -- Display the result
    -- preds_hm:mul(4) -- Change to input scale
    -- local dispImg = drawOutput(inp, hm, preds_hm[1])
    -- w = image.display{image=dispImg,win=w}
    -- print(preds_img)
    -- sys.sleep(3)

    collectgarbage()
end

print("Saving predictions...")
-- Save predictions
local predFile = hdf5.open('preds/he_test_cropped.h5', 'w')

for i=1,nbSamples do
    predFile:write(filelist[i-1], preds[i])
end

-- predFile:write('preds', preds)
-- predFile:write('paths', paths)
predFile:close()

-- --------------------------------------------------------------------------------
-- -- Evaluation code
-- --------------------------------------------------------------------------------

-- if arg[1] == 'eval' then
--     -- Calculate distances given each set of predictions
--     local labels = {'valid-example','valid-ours'}
--     local dists = {}
--     for i = 1,#labels do
--         local predFile = hdf5.open('preds/' .. labels[i] .. '.h5','r')
--         local preds = predFile:read('preds'):all()
--         table.insert(dists,calcDists(preds, a.part, a.normalize))
--     end

--     require 'gnuplot'
--     gnuplot.raw('set bmargin 1')
--     gnuplot.raw('set lmargin 3.2')
--     gnuplot.raw('set rmargin 2')    
--     gnuplot.raw('set multiplot layout 2,3 title "MPII Validation Set Performance (PCKh)"')
--     gnuplot.raw('set xtics font ",6"')
--     gnuplot.raw('set ytics font ",6"')
--     displayPCK(dists, {9,10}, labels, 'Head')
--     displayPCK(dists, {2,5}, labels, 'Knee')
--     displayPCK(dists, {1,6}, labels, 'Ankle')
--     gnuplot.raw('set tmargin 2.5')
--     gnuplot.raw('set bmargin 1.5')
--     displayPCK(dists, {13,14}, labels, 'Shoulder')
--     displayPCK(dists, {12,15}, labels, 'Elbow')
--     displayPCK(dists, {11,16}, labels, 'Wrist', true)
--     gnuplot.raw('unset multiplot')
-- end
