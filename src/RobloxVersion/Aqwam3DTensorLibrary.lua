--[[

	--------------------------------------------------------------------

	Version 0.0.0

	Aqwam's 3D Tensor Library (TensorL3D)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our terms and conditions at:
	
	https://github.com/AqwamCreates/TensorL3D/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT WITHOUT PERMISSION!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary3D = {}

local function create3DTensor(dimensionArray, initialValue)
	
	local result = {}

	for dimension1 = 1, dimensionArray[1], 1 do

		result[dimension1] =  {}

		for dimension2 = 1, dimensionArray[2], 1 do

			result[dimension1][dimension2] = table.create(dimensionArray[3], initialValue)

		end

	end
	
	return result
	
end

local function create3DTensorFromFunction(dimensionArray, functionToApply)

	local result = {}

	for dimension1 = 1, dimensionArray[1], 1 do

		result[dimension1] =  {}

		for dimension2 = 1, dimensionArray[2], 1 do
			
			result[dimension1][dimension2] =  {}
			
			for dimension3 = 1, dimensionArray[3], 1 do
				
				result[dimension1][dimension2][dimension3] = functionToApply(dimension1, dimension2, dimension3)
				
			end

		end

	end

	return result

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function onBroadcastError(tensor1, tensor2)

	local errorMessage = "Unable To Broadcast. \n" .. "Tensor 1 Size: " .. "(" .. #tensor1 .. ", " .. #tensor1[1] .. ", " .. #tensor1[1][1] .. ") \n" .. "Tensor 2 Size: " .. "(" .. #tensor2 .. ", " .. #tensor2[1] .. ", " .. #tensor2[1][1] .. ") \n"

	error(errorMessage)

end

local function checkIfCanBroadcast(tensor1, tensor2)

	local tensor1Depth = #tensor1
	local tensor2Depth = #tensor2

	local tensor1Rows = #tensor1[1]
	local tensor2Rows = #tensor2[1]

	local tensor1Columns = #tensor1[1][1]
	local tensor2Columns = #tensor2[1][1]

	local isTensor1Broadcasted
	local isTensor2Broadcasted

	local hasSameRowSize = (tensor1Rows == tensor2Rows)
	local hasSameColumnSize = (tensor1Columns == tensor2Columns)
	local hasSameDepth = (tensor1Depth == tensor2Depth)

	local hasSameDimension = hasSameRowSize and hasSameColumnSize and hasSameDepth

	local isTensor1LargerInOneDimension = ((tensor1Depth > 1) and hasSameRowSize and hasSameColumnSize and (tensor2Depth == 1)) or
		((tensor1Rows > 1) and hasSameColumnSize and hasSameDepth and (tensor2Rows == 1)) or
		((tensor1Columns > 1) and hasSameRowSize and hasSameDepth and (tensor2Columns == 1))

	local isTensor2LargerInOneDimension = ((tensor2Depth > 1) and hasSameRowSize and hasSameColumnSize and (tensor1Depth == 1)) or
		((tensor2Rows > 1) and hasSameColumnSize and hasSameDepth and (tensor1Rows == 1)) or
		((tensor2Columns > 1) and hasSameRowSize and hasSameDepth and (tensor1Columns == 1))

	local isTensor1Scalar = (tensor1Depth == 1) and (tensor1Rows == 1) and (tensor1Columns == 1)
	local isTensor2Scalar = (tensor2Depth == 1) and (tensor2Rows == 1) and (tensor2Columns == 1)

	local isTensor1Larger = ((tensor1Depth > tensor2Depth) or (tensor1Rows > tensor2Rows) or (tensor1Columns > tensor2Columns)) and not ((tensor1Depth < tensor2Depth) or (tensor1Rows < tensor2Rows) or (tensor1Columns < tensor2Columns))
	local isTensor2Larger = ((tensor2Depth > tensor1Depth) or (tensor2Rows > tensor1Rows) or (tensor2Columns > tensor1Columns)) and not ((tensor2Depth < tensor1Depth) or (tensor2Rows < tensor1Rows) or (tensor2Columns < tensor1Columns))

	if (hasSameDimension) then

		isTensor1Broadcasted = false
		isTensor2Broadcasted = false

	elseif (isTensor2LargerInOneDimension) or (isTensor2Larger and isTensor1Scalar) then

		isTensor1Broadcasted = true
		isTensor2Broadcasted = false

	elseif (isTensor1LargerInOneDimension) or (isTensor1Larger and isTensor2Scalar) then

		isTensor1Broadcasted = false
		isTensor2Broadcasted = true

	else

		onBroadcastError(tensor1, tensor2)

	end

	return isTensor1Broadcasted, isTensor2Broadcasted

end

local function expandTensor(tensor, targetDimensionArray)
	
	local targetDepthSize = targetDimensionArray[1]
	
	local targetRowSize = targetDimensionArray[2]
	
	local targetColumnSize = targetDimensionArray[3]

	local isDepthSizeEqualToOne = (#tensor == 1)

	local isRowSizeEqualToOne = (#tensor[1] == 1)

	local isColumnSizeEqualToOne = (#tensor[1][1] == 1)

	local result = {}

	if (isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize, 1 do

					result[i][j][k] = tensor[1][1][1]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize,1 do

					result[i][j][k] = tensor[i][1][1]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize, 1 do

					result[i][j][k] = tensor[i][j][1]

				end

			end

		end

	elseif (isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize, 1 do

					result[i][j][k] = tensor[1][j][1]

				end

			end

		end

	elseif (isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize, 1 do

					result[i][j][k] = tensor[1][1][k]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize, 1 do

					result[i][j][k] = tensor[i][1][k]

				end

			end

		end

	elseif (isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize, 1 do

			result[i] = {}

			for j = 1, targetRowSize, 1 do

				result[i][j] = {}

				for k = 1, targetColumnSize, 1 do

					result[i][j][k] = tensor[1][j][k]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		result = tensor

	end

	return result

end

local function broadcastTensorsIfDifferentSizes(tensor1, tensor2)

	local isTensor1Broadcasted, isTensor2Broadcasted = checkIfCanBroadcast(tensor1, tensor2)

	if (isTensor1Broadcasted) then
		
		local targetDimensionArray = AqwamTensorLibrary3D:getSize(tensor2)

		tensor1 = expandTensor(tensor1, targetDimensionArray)

	elseif (isTensor2Broadcasted) then
		
		local targetDimensionArray = AqwamTensorLibrary3D:getSize(tensor1)

		tensor2 = expandTensor(tensor2, targetDimensionArray)

	end

	return tensor1, tensor2

end

local function applyFunction(functionToApply, tensor1, tensor2)
	
	local result = {}
	
	for dimension1 = 1, #tensor1, 1 do
		
		result[dimension1] = {}

		for dimension2 = 1, #tensor1[dimension1], 1 do
			
			result[dimension1][dimension2] = {}

			for dimension3 = 1, #tensor1[dimension1][dimension2], 1 do

				result[dimension1][dimension2][dimension3] = functionToApply(tensor1[dimension1][dimension2][dimension3], tensor2[dimension1][dimension2][dimension3]) 

			end

		end

	end
	
	return result
	
end

local function sum(tensor, dimension)
	
	local dimensionArray = AqwamTensorLibrary3D:getSize(tensor)

	local newDimensionArray = deepCopyTable(dimensionArray)

	if (dimension) then

		if (dimension <= 0) or (dimension >= 4) then error("The dimension must be between 1 and 3.") end

		newDimensionArray[dimension] = 1

	end

	local result = (not dimension and 0) or AqwamTensorLibrary3D:createTensor(newDimensionArray, 0)

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			for dimension3 = 1, dimensionArray[3], 1 do

				if (dimension == nil) then

					result += tensor[dimension1][dimension2][dimension3]

				elseif (dimension == 1) then

					result[1][dimension2][dimension3] += tensor[dimension1][dimension2][dimension3]	

				elseif (dimension == 2) then

					result[dimension1][1][dimension3] += tensor[dimension1][dimension2][dimension3]

				elseif (dimension == 3) then

					result[dimension1][dimension2][1] += tensor[dimension1][dimension2][dimension3]

				else

					error("Invalid dimension.")

				end 

			end

		end	

	end
	
	return result
	
end

local function is3DTensor(tensor)

	local isTensor = pcall(function() local _ = tensor[1][1][1] end)

	return isTensor

end

local function convertValueTo3DTensor(value)

	if is3DTensor(value) then return value end

	if (type(value) ~= "number") then error("Cannot convert value into 3D tensor.") end

	return {{{value}}}

end

local function isDimensionArrayEqual(dimensionArray, otherDimensionArray)
	
	for index, _ in ipairs(dimensionArray) do if (dimensionArray[index] ~= otherDimensionArray[index]) then return false end end
	
	return true
	
end

local function throwErrorIfValueIsNot3DTensor(otherTensor)

	if not is3DTensor(otherTensor) then error("The value is not a 3D tensor.") end

end

local function throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)

	if (#dimensionArray ~= 3) then error("The length of dimension array is not equal to 3.") end

end

local function throwErrorIfDimensionArrayIsNotEqual(dimensionArray, otherDimensionArray)

	if not isDimensionArrayEqual(dimensionArray, otherDimensionArray) then error("The values of dimension arrays are not equal.") end

end

local function applyFunctionOnMultiple3DTensors(functionToApply, ...)
	
	local tensorArray = {...}
	
	local result = deepCopyTable(tensorArray[1])
	
	result = convertValueTo3DTensor(result)
	
	for i = 2, #tensorArray, 1 do
		
		local otherTensor = tensorArray[i]

		otherTensor = convertValueTo3DTensor(otherTensor)
		
		result, otherTensor = broadcastTensorsIfDifferentSizes(result, otherTensor)
		
		result = applyFunction(functionToApply, result, otherTensor)
		
	end
	
	return result
	
end

function AqwamTensorLibrary3D:createTensor(dimensionArray, initialValue)

	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)

	initialValue = initialValue or 0

	return create3DTensor(dimensionArray, initialValue)

end

function AqwamTensorLibrary3D:createTensorFromFunction(dimensionArray, functionToApply)
	
	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)
	
	if (type(functionToApply) == "nil") then error("No function.") end

	return create3DTensorFromFunction(dimensionArray, functionToApply)
	
end

function AqwamTensorLibrary3D:createIdentityTensor(dimensionArray)

	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)

	local newTensor = {}

	for i = 1, dimensionArray[1], 1 do

		newTensor[i] = {}

		for j = 1, dimensionArray[2], 1 do

			newTensor[i][j] = {}

			for k = 1, dimensionArray[3], 1 do

				local areEqual = (i == j) and (j == k)

				newTensor[i][j][k] = (areEqual and 1) or 0

			end

		end

	end

	return newTensor

end

function AqwamTensorLibrary3D:createRandomUniformTensor(dimensionArray)

	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)

	local newTensor = {}

	for i = 1, dimensionArray[1], 1 do

		newTensor[i] = {}

		for j = 1, dimensionArray[2], 1 do

			newTensor[i][j] = {}

			for k = 1, dimensionArray[3], 1 do
				
				newTensor[i][j][k] = math.random()

			end

		end

	end

	return newTensor

end

function AqwamTensorLibrary3D:createRandomNormalTensor(dimensionArray, mean, standardDeviation)

	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	local newTensor = {}

	for i = 1, dimensionArray[1], 1 do

		newTensor[i] = {}

		for j = 1, dimensionArray[2], 1 do

			newTensor[i][j] = {}

			for k = 1, dimensionArray[3], 1 do

				local randomNumber1 = math.random()

				local randomNumber2 = math.random()

				local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

				newTensor[i][j][k] = (zScore * standardDeviation) + mean

			end

		end

	end

	return newTensor

end

function AqwamTensorLibrary3D:expand(tensor, dimensionArray)
	
	return expandTensor(tensor, dimensionArray)

end

function AqwamTensorLibrary3D:getSize(tensor)
	
	throwErrorIfValueIsNot3DTensor(tensor)
	
	return {#tensor, #tensor[1], #tensor[1][1]}
	
end

function AqwamTensorLibrary3D:generateTensor2DString(tensor2D)

	if tensor2D == nil then return "" end

	local numberOfRows = #tensor2D

	local numberOfColumns = #tensor2D[1]

	local columnWidths = {}

	for column = 1, numberOfColumns do

		local maxWidth = 0

		for row = 1, numberOfRows do

			local cellWidth = string.len(tostring(tensor2D[row][column]))

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[column] = maxWidth

	end

	local text = ""

	for row = 1, numberOfRows do

		text = text .. "{"

		for column = 1, numberOfColumns do

			local cellValue = tensor2D[row][column]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = columnWidths[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText
		end

		text = text .. " }\n"

	end

	return text

end

function AqwamTensorLibrary3D:generateTensorString(tensor)
	
	local text = "\n\n{\n\n"

	local generatedText

	for index = 1, #tensor, 1 do

		generatedText = AqwamTensorLibrary3D:generateTensor2DString(tensor[index])

		text = text .. generatedText .. "\n"

	end

	text = text .. "}\n\n"
	
	return text
	
end

function AqwamTensorLibrary3D:printTensor(tensor)
	
	throwErrorIfValueIsNot3DTensor(tensor)

	local text = AqwamTensorLibrary3D:generateTensorString(tensor)
	
	print(text)
	
end

function AqwamTensorLibrary3D:transpose(tensor, dimensionIndexArray)

	if (#dimensionIndexArray ~= 2) then error("The length of dimension index array is not equal to 2.") end

	local dimension1 = dimensionIndexArray[1]

	local dimension2 = dimensionIndexArray[2]

	if (type(dimension1) ~= "number") or (type(dimension2) ~= "number") then error("Dimensions are not numbers.") end

	if (dimension1 <= 0) or (dimension1 >= 4) or (dimension2 <= 0) or (dimension2 >= 4) or (dimension1 == dimension2) then

		error("Invalid dimensions for transpose.")

	end

	local newDimensionArray = AqwamTensorLibrary3D:getSize(tensor)

	newDimensionArray[dimension1], newDimensionArray[dimension2] = newDimensionArray[dimension2], newDimensionArray[dimension1]

	local newTensor = AqwamTensorLibrary3D:createTensor(newDimensionArray, true)

	if (table.find(dimensionIndexArray, 1)) and (table.find(dimensionIndexArray, 2)) then

		for i = 1, newDimensionArray[1] do

			for j = 1, newDimensionArray[2] do

				for k = 1, newDimensionArray[3] do

					newTensor[i][j][k] = tensor[j][i][k]

				end

			end

		end

	elseif (table.find(dimensionIndexArray, 1)) and (table.find(dimensionIndexArray, 3)) then

		for i = 1, newDimensionArray[1] do

			for j = 1, newDimensionArray[2] do

				for k = 1, newDimensionArray[3] do

					newTensor[i][j][k] = tensor[k][j][i]

				end

			end

		end

	elseif (table.find(dimensionIndexArray, 2)) and (table.find(dimensionIndexArray, 3)) then

		for i = 1, newDimensionArray[1] do

			for j = 1, newDimensionArray[2] do

				for k = 1, newDimensionArray[3] do

					newTensor[i][j][k] = tensor[i][k][j]

				end

			end

		end

	end

	return newTensor

end

function AqwamTensorLibrary3D:isSameMatrix(tensor1, tensor2)
	
	local dimensionSizeArray1 = AqwamTensorLibrary3D:getSize(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary3D:getSize(tensor2)

	for i, size in ipairs(dimensionSizeArray1) do if (size ~= dimensionSizeArray2[i]) then return false end end

	for i = 1, dimensionSizeArray1[1], 1 do

		for j = 1, dimensionSizeArray1[2], 1 do

			for k = 1, dimensionSizeArray1[3], 1 do

				for l = 1, dimensionSizeArray1[3], 1 do

					if (tensor1[i][j][k][l] ~= tensor2[i][j][k][l]) then return false end

				end

			end

		end

	end

	return true
	
end

function AqwamTensorLibrary3D:isEqualTo(tensor1, tensor2)
	
	throwErrorIfValueIsNot3DTensor(tensor1)
	
	throwErrorIfValueIsNot3DTensor(tensor2)

	local functionToApply = function(a, b) return (a == b) end

	local result = applyFunction(functionToApply, tensor1, tensor2)

	return result

end

function AqwamTensorLibrary3D:isGreaterThan(tensor1, tensor2)

	throwErrorIfValueIsNot3DTensor(tensor1)

	throwErrorIfValueIsNot3DTensor(tensor2)

	local functionToApply = function(a, b) return (a > b) end

	local result = applyFunction(functionToApply, tensor1, tensor2)

	return result

end

function AqwamTensorLibrary3D:isGreaterOrEqualTo(tensor1, tensor2)

	throwErrorIfValueIsNot3DTensor(tensor1)

	throwErrorIfValueIsNot3DTensor(tensor2)

	local functionToApply = function(a, b) return (a >= b) end

	local result = applyFunction(functionToApply, tensor1, tensor2)

	return result

end

function AqwamTensorLibrary3D:isLessThan(tensor1, tensor2)

	throwErrorIfValueIsNot3DTensor(tensor1)

	throwErrorIfValueIsNot3DTensor(tensor2)
	
	local functionToApply = function(a, b) return (a < b) end

	local result = applyFunction(functionToApply, tensor1, tensor2)

	return result

end

function AqwamTensorLibrary3D:isLessOrEqualTo(tensor1, tensor2)

	throwErrorIfValueIsNot3DTensor(tensor1)

	throwErrorIfValueIsNot3DTensor(tensor2)

	local functionToApply = function(a, b) return (a <= b) end

	local result = applyFunction(functionToApply, tensor1, tensor2)

	return result

end

function AqwamTensorLibrary3D:sum(tensor, dimension)
	
	return sum(tensor, dimension)
	
end

function AqwamTensorLibrary3D:concatenate(tensor1, tensor2, dimension)

	if (dimension <= 0) or (dimension >= 4) then error("The dimension must be between 1 and 3.") end

	tensor1 = convertValueTo3DTensor(tensor1)
	
	tensor2 = convertValueTo3DTensor(tensor2)
	
	throwErrorIfValueIsNot3DTensor(tensor1)

	throwErrorIfValueIsNot3DTensor(tensor2)

	local dimensionSizeArray1 = AqwamTensorLibrary3D:getSize(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary3D:getSize(tensor2)

	local newDimensionArray = {}

	for dimensionIndex = 1, 3, 1 do

		if (dimensionIndex == dimension) then continue end

		if (dimensionSizeArray1[dimensionIndex] ~= dimensionSizeArray2[dimensionIndex]) then error("The tensors do not contain equal dimension values at dimension " .. dimensionIndex .. ".") end

	end

	for dimensionIndex = 1, 3, 1 do

		local dimensionSize = dimensionSizeArray1[dimensionIndex]

		if (dimensionIndex == dimension) then

			dimensionSize = dimensionSize + dimensionSizeArray2[dimensionIndex]

		end

		table.insert(newDimensionArray, dimensionSize)

	end

	local newTensor = AqwamTensorLibrary3D:create(newDimensionArray, true)

	for i = 1, dimensionSizeArray1[1], 1 do

		for j = 1, dimensionSizeArray1[2], 1 do

			for k = 1, dimensionSizeArray1[3],1 do

				newTensor[i][j][k] = tensor1[i][j][k]

			end

		end

	end

	if (dimension == 1) then

		local newDimensionHalfSize = dimensionSizeArray1[1]

		for i = 1, dimensionSizeArray2[1], 1 do

			for j = 1, dimensionSizeArray2[2], 1 do

				for k = 1, dimensionSizeArray2[3],1 do

					newTensor[newDimensionHalfSize + i][j][k] = tensor2[i][j][k]

				end

			end

		end

	elseif (dimension == 2) then

		local newDimensionHalfSize = dimensionSizeArray1[2]

		for i = 1, dimensionSizeArray2[1], 1 do

			for j = 1, dimensionSizeArray2[2], 1 do

				for k = 1, dimensionSizeArray2[3],1 do

					newTensor[i][newDimensionHalfSize + j][k] = tensor2[i][j][k]

				end

			end

		end

	elseif (dimension == 3) then

		local newDimensionHalfSize = dimensionSizeArray1[3]

		for i = 1, dimensionSizeArray2[1], 1 do

			for j = 1, dimensionSizeArray2[2], 1 do

				for k = 1, dimensionSizeArray2[3],1 do

					newTensor[i][j][newDimensionHalfSize + k] = tensor2[i][j][k]

				end

			end

		end

	end

	return newTensor

end

function AqwamTensorLibrary3D:dotProduct(tensor1, tensor2) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc
	
	tensor1 = convertValueTo3DTensor(tensor1)
	
	tensor2 = convertValueTo3DTensor(tensor2)
	
	throwErrorIfValueIsNot3DTensor(tensor1)
	
	throwErrorIfValueIsNot3DTensor(tensor2)
	
	local dimensionSizeArray1 = AqwamTensorLibrary3D:getSize(tensor1)
	
	local dimensionSizeArray2 = AqwamTensorLibrary3D:getSize(tensor2)
	
	if (dimensionSizeArray1[1] ~= dimensionSizeArray2[1]) then error("The tensors do not contain equal dimension values at dimension 1.") end

	if (dimensionSizeArray1[3] ~= dimensionSizeArray2[2]) then error("The size of the dimension 3 of the first tensor is not equal to the size of dimension 2 of the second tensor.") end

	local newTensor = create3DTensor({dimensionSizeArray1[1], dimensionSizeArray1[2], dimensionSizeArray2[3]}, true)
	
	for i = 1, dimensionSizeArray1[1], 1 do

		for j = 1, dimensionSizeArray1[2], 1 do

			for k = 1, dimensionSizeArray2[3], 1 do

				local sum = 0

				for l = 1, dimensionSizeArray1[3] do sum = sum + (tensor1[i][j][l] * tensor2[i][l][k]) end

				newTensor[i][j][k] = sum

			end

		end

	end
	
	return newTensor
	
end

function AqwamTensorLibrary3D:innerProduct(...)
	
	local tensorArray = {...}
	
	local result = tensorArray[1]
	
	result = convertValueTo3DTensor(result)
	
	local functionToApply = function(a, b) return (a * b) end
	
	throwErrorIfValueIsNot3DTensor(result)
	
	for i = 2, #tensorArray, 1 do
		
		local otherTensor = tensorArray[i]
		
		otherTensor = convertValueTo3DTensor(otherTensor)
		
		throwErrorIfValueIsNot3DTensor(otherTensor)
		
		result = applyFunction(functionToApply, result, otherTensor)
		
		result = AqwamTensorLibrary3D:sum(result, 1)

		result = AqwamTensorLibrary3D:sum(result, 2)
		
		result = AqwamTensorLibrary3D:sum(result, 3)
		
	end
	
	return result[1][1][1]

end

function AqwamTensorLibrary3D:copy(tensor)
	
	return deepCopyTable(tensor)
	
end

function AqwamTensorLibrary3D:applyFunction(functionToApply, ...)

	local tensorValues

	local tensorsArray = {...}
	
	local dimensionSizeArray = AqwamTensorLibrary3D:getSize(tensorsArray[1])
	
	local result = create3DTensor(dimensionSizeArray)

	for dimension1 = 1, dimensionSizeArray[1], 1 do

		for dimension2 = 1, dimensionSizeArray[2], 1 do

			for dimension3 = 1, dimensionSizeArray[3], 1 do
				
				tensorValues = {}
				
				for i, value in ipairs(tensorsArray) do
					
					if (type(value) == "number") then
						
						table.insert(tensorValues, value)
						
					else
						
						table.insert(tensorValues, value[dimension1][dimension2][dimension3])
						
					end
					
				end
				
				result[dimension1][dimension2][dimension3] = functionToApply(table.unpack(tensorValues))
				
			end
			
		end	

	end

	return result

end

function AqwamTensorLibrary3D:add(...)
	
	local functionToApply = function(a, b) return (a + b) end

	return applyFunctionOnMultiple3DTensors(functionToApply, ...)
	
end

function AqwamTensorLibrary3D:subtract(...)

	local functionToApply = function(a, b) return (a - b) end

	return applyFunctionOnMultiple3DTensors(functionToApply, ...)
	
end

function AqwamTensorLibrary3D:multiply(...)
	
	local functionToApply = function(a, b) return (a * b) end

	return applyFunctionOnMultiple3DTensors(functionToApply, ...)
	
end

function AqwamTensorLibrary3D:divide(...)

	local functionToApply = function(a, b) return (a / b) end

	return applyFunctionOnMultiple3DTensors(functionToApply, ...)
	
end

function AqwamTensorLibrary3D:logarithm(...)

	return applyFunctionOnMultiple3DTensors(math.log, ...)
	
end

function AqwamTensorLibrary3D:exponent(...)

	return applyFunctionOnMultiple3DTensors(math.exp, ...)

end

function AqwamTensorLibrary3D:power(...)

	return applyFunctionOnMultiple3DTensors(math.pow, ...)

end

function AqwamTensorLibrary3D:unaryMinus(tensor)
	
	local result = {}
	
	local dimensionSizeArray = AqwamTensorLibrary3D:getSize(tensor)

	for dimension1 = 1, dimensionSizeArray[1], 1 do

		result[dimension1] = {}

		for dimension2 = 1, dimensionSizeArray[2], 1 do

			result[dimension1][dimension2] = {}

			for dimension3 = 1, dimensionSizeArray[3], 1 do

				result[dimension1][dimension2][dimension3] = -tensor[dimension1][dimension2][dimension3]

			end

		end

	end
	
	return result
	
end

return AqwamTensorLibrary3D
