local BaseShapeTransformationBlock = require(script.Parent.BaseShapeTransformationBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

Flatten = {}

Flatten.__index = Flatten

setmetatable(Flatten, BaseShapeTransformationBlock)

function Flatten.new()

	local NewFlatten = BaseShapeTransformationBlock.new()

	setmetatable(NewFlatten, Flatten)

	NewFlatten:setName("Flatten")
	
	NewFlatten:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewFlatten:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:flatten(inputTensorArray[1])
	
	end)

	NewFlatten:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local dimensionSizeArray = AqwamTensorLibrary:getSize(inputTensorArray[1])
		
		local firstDerivativeTensor = AqwamTensorLibrary:reshape(initialPartialFirstDerivativeTensor, dimensionSizeArray)
		
		print(dimensionSizeArray)
		
		return {firstDerivativeTensor}
		
	end)

	return NewFlatten

end

return Flatten
